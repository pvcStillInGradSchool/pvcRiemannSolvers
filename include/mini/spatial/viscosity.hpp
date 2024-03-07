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
  using Projection = typename Base::Projection;
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
  static FluxMatrix GetDiffusiveFluxMatrix(const Cell &cell, int q) {
    const auto &projection = cell.projection();
    const auto &value = projection.GetValue(q);
    FluxMatrix flux_matrix; flux_matrix.setZero();
    const auto &gradient = projection.GetGlobalGradient(q);
    Riemann::MinusViscousFlux(value, gradient, &flux_matrix);
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
        // Write the residual column intto the matrix:
        matrix.col(c) = residual.row(0);
        for (int r = 1; r < Cell::K; ++r) {
          assert((residual.row(r) - residual.row(0)).squaredNorm() == 0);
        }
      }
    }
    return matrices;
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
  void ApplySmartBoundary(Column *residual) const override {
    base().ApplySmartBoundary(residual);
  }
  void ApplyInviscidWall(Column *residual) const override {
    base().ApplyInviscidWall(residual);
  }
  void ApplySubsonicInlet(Column *residual) const override {
    base().ApplySubsonicInlet(residual);
  }
  void ApplySubsonicOutlet(Column *residual) const override {
    base().ApplySubsonicOutlet(residual);
  }
  void ApplySupersonicInlet(Column *residual) const override {
    base().ApplySupersonicInlet(residual);
  }
  void ApplySupersonicOutlet(Column *residual) const override {
    base().ApplySupersonicOutlet(residual);
  }
};

}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_VISCOSITY_HPP_
