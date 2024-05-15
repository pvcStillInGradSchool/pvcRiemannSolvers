// Copyright 2024 PEI Weicheng
#ifndef MINI_SPATIAL_VISCOSITY_HPP_
#define MINI_SPATIAL_VISCOSITY_HPP_

#include <concepts>

#include <cassert>
#include <fstream>
#include <functional>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "mini/algebra/eigen.hpp"
#include "mini/riemann/concept.hpp"
#include "mini/temporal/ode.hpp"
#include "mini/spatial/fem.hpp"
#include "mini/riemann/concept.hpp"
#include "mini/riemann/diffusive/linear.hpp"
#include "mini/riemann/diffusive/direct_dg.hpp"

namespace mini {
namespace spatial {

template <typename Part>
class EnergyBasedViscosity : public FiniteElement<Part> {
 public:
  using Base = FiniteElement<Part>;
  using Riemann = typename Base::Riemann;
  using Scalar = typename Base::Scalar;
  using Face = typename Base::Face;
  using Cell = typename Base::Cell;
  using Global = typename Base::Global;
  using Polynomial = typename Base::Polynomial;
  using Integrator = typename Polynomial::Integrator;
  using Local = typename Integrator::Local;
  using Coeff = typename Base::Coeff;
  using Value = typename Base::Value;
  using Temporal = typename Base::Temporal;
  using Column = typename Base::Column;
  using FluxMatrix = typename Base::FluxMatrix;
  using CellToFlux = typename Base::CellToFlux;
  using DampingMatrix = algebra::Matrix<Scalar, Cell::N, Cell::N>;

 private:
  Base *base_ptr_;

  using Diffusion = mini::riemann::diffusive::Isotropic<Scalar, Cell::K>;
  using DiffusionRiemann = mini::riemann::diffusive::DirectDG<Diffusion>;

  static FluxMatrix GetDiffusiveFluxMatrix(const Cell &cell, int q) {
    const auto &projection = cell.projection();
    const auto &value = projection.GetValue(q);
    FluxMatrix flux_matrix; flux_matrix.setZero();
    const auto &gradient = projection.GetGlobalGradient(q);
    DiffusionRiemann::MinusViscousFlux(value, gradient, &flux_matrix);
    return flux_matrix;
  }

 public:
  explicit EnergyBasedViscosity(Base *base_ptr)
      : Base(base_ptr->part_ptr()), base_ptr_(base_ptr) {
  }
  EnergyBasedViscosity(const EnergyBasedViscosity &) = default;
  EnergyBasedViscosity &operator=(const EnergyBasedViscosity &) = default;
  EnergyBasedViscosity(EnergyBasedViscosity &&) noexcept = default;
  EnergyBasedViscosity &operator=(EnergyBasedViscosity &&) noexcept = default;
  ~EnergyBasedViscosity() noexcept = default;

  Base const &base() const {
    return *base_ptr_;
  }
  Part const &part() const {
    return base().part();
  }

  std::string name() const override {
    return base().name() + "EnergyBasedViscosity";
  }

 protected:  // data for generating artificial viscosity
  std::vector<DampingMatrix> damping_matrices_;

  // [i_cell][i_neighbor][i_node]
  std::vector<std::vector<std::array<Local, Cell::N>>> local_on_neighbors_;

 public:  // methods for generating artificial viscosity
  std::vector<DampingMatrix> BuildDampingMatrices() const {
    auto matrices = std::vector<DampingMatrix>(part().CountLocalCells());
    Diffusion::SetDiffusionCoefficient(1.0);
    for (Cell *cell_ptr: base_ptr_->part_ptr()->GetLocalCellPointers()) {
      // Nullify all its neighbors' coeffs:
      for (Cell *neighbor : cell_ptr->adj_cells_) {
        neighbor->projection().coeff().setZero();
      }
      // Build the damping matrix column by column:
      auto &matrix = matrices.at(cell_ptr->id());
      auto &solution = cell_ptr->projection().coeff();
      solution.setZero();
      for (int c = 0; c < Cell::N; ++c) {
        solution.col(c).setOnes();
        if (c > 0) {
          solution.col(c - 1).setZero();
        }
        // Build the element-wise residual column:
        Coeff residual; residual.setZero();
        base().AddFluxDivergence(GetDiffusiveFluxMatrix, *cell_ptr,
            residual.data());
        // Write the residual column into the matrix:
        matrix.col(c) = residual.row(0);
        for (int r = 1; r < Cell::K; ++r) {
          assert((residual.row(r) - residual.row(0)).squaredNorm() == 0);
        }
      }
    }
    return matrices;
  }

  std::vector<std::vector<std::array<Local, Cell::N>>> BuildCoordinates() const {
    std::vector<std::vector<std::array<Local, Cell::N>>> local_on_neighbors;
    local_on_neighbors.reserve(part().CountLocalCells());
    for (Cell *cell_ptr : base_ptr_->part_ptr()->GetLocalCellPointers()) {
      auto &local_on_neighbors_of_cell_i = local_on_neighbors.emplace_back();
      int n_neighbor = cell_ptr->adj_cells_.size();
      local_on_neighbors_of_cell_i.reserve(n_neighbor);
      for (int i_neighbor = 0; i_neighbor < n_neighbor; ++i_neighbor) {
        auto &local_on_neighbor_i = local_on_neighbors_of_cell_i.emplace_back();
        auto const &coordinate_i = cell_ptr->adj_cells_[i_neighbor]->coordinate();
        for (int i_node = 0; i_node < Cell::N; ++i_node) {
          auto &xyz = cell_ptr->integrator().GetGlobal(i_node);
          std::printf("%d %ld %d %d\n", part().mpi_rank(), cell_ptr->id(), i_neighbor, i_node);
          local_on_neighbor_i[i_node] = coordinate_i.GlobalToLocal(xyz);
        }
      }
    }
    assert(local_on_neighbors.size() == part().CountLocalCells());
    return local_on_neighbors;
  }

 public:  // override virtual methods defined in Base
  Column GetResidualColumn() const override {
    return base().GetResidualColumn();
  }
  Column GetSolutionColumn() const override {
    return base().GetSolutionColumn();
  }
  void SetSolutionColumn(Column const &column) override {
    base_ptr_->SetSolutionColumn(column);
  }
  void SetTime(double t_curr) override {
    base_ptr_->SetTime(t_curr);
  }

 protected:  // override virtual methods defined in Base
  void AddFluxDivergence(CellToFlux cell_to_flux, Cell const &cell,
      Scalar *data) const override {
    base().AddFluxDivergence(cell_to_flux, cell, data);
  }
  void AddFluxDivergence(CellToFlux cell_to_flux,
      Column *residual) const override {
    base().AddFluxDivergence(cell_to_flux, residual);
  }
  void AddFluxOnGhostFaces(Column *residual) const override {
    base().AddFluxOnGhostFaces(residual);
  }
  void AddFluxOnLocalFaces(Column *residual) const override {
    base().AddFluxOnLocalFaces(residual);
  }
  void AddFluxOnSmartBoundaries(Column *residual) const override {
    base().AddFluxOnSmartBoundaries(residual);
  }
  void AddFluxOnInviscidWalls(Column *residual) const override {
    base().AddFluxOnInviscidWalls(residual);
  }
  void AddFluxOnSubsonicInlets(Column *residual) const override {
    base().AddFluxOnSubsonicInlets(residual);
  }
  void AddFluxOnSubsonicOutlets(Column *residual) const override {
    base().AddFluxOnSubsonicOutlets(residual);
  }
  void AddFluxOnSupersonicInlets(Column *residual) const override {
    base().AddFluxOnSupersonicInlets(residual);
  }
  void AddFluxOnSupersonicOutlets(Column *residual) const override {
    base().AddFluxOnSupersonicOutlets(residual);
  }
};

}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_VISCOSITY_HPP_
