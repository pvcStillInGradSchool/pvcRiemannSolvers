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

  // TODO(PVC): avoid overriding Base::Riemann
  using Diffusion = mini::riemann::diffusive::Isotropic<Scalar, Cell::K>;
  using DiffusionRiemann = mini::riemann::diffusive::DirectDG<Diffusion>;

  static FluxMatrix GetDiffusiveFluxMatrix(const Cell &cell, int q) {
    const auto &polynomial = cell.polynomial();
    const auto &value = polynomial.GetValue(q);
    FluxMatrix flux_matrix; flux_matrix.setZero();
    const auto &gradient = polynomial.GetGlobalGradient(q);
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

 public:  // methods for generating artificial viscosity
  std::vector<DampingMatrix> BuildDampingMatrices() const {
    auto matrices = std::vector<DampingMatrix>(part().CountLocalCells());
    Diffusion::SetDiffusionCoefficient(1.0);
    for (Cell *cell_ptr: base_ptr_->part_ptr()->GetLocalCellPointers()) {
      // Nullify all its neighbors' coeffs:
      for (Cell *neighbor : cell_ptr->adj_cells_) {
        neighbor->polynomial().coeff().setZero();
      }
      // Build the damping matrix column by column:
      auto &matrix = matrices.at(cell_ptr->id());
      auto &solution = cell_ptr->polynomial().coeff();
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
          assert(residual.row(r) == residual.row(0));
        }
      }
#ifndef NDEBUG
      Coeff residual; residual.setZero();
      solution.setOnes();
      base().AddFluxDivergence(GetDiffusiveFluxMatrix, *cell_ptr,
          residual.data());
      for (int k = 0; k < Cell::K; ++k) {
        auto const &residual_col = residual.row(k).transpose();
        auto const &solution_col = solution.row(k).transpose();
        assert((residual_col - matrix * solution_col).norm() < 1e-10);
      }
#endif
      // Scale the damping matrix by local weight over Jacobian:
      assert(cell_ptr->polynomial().kLocal);
      for (int r = 0; r < Cell::N; ++r) {
        Scalar scale
            = cell_ptr->integrator().GetLocalWeight(r)
            / cell_ptr->integrator().GetJacobianDeterminant(r);
        matrix.row(r) *= scale;
      }
    }
    return matrices;
  }

  std::vector<std::array<Value, Cell::N>> BuildValueJumps() const {
    std::vector<std::array<Value, Cell::N>> value_jumps;
    value_jumps.reserve(part().CountLocalCells());
    for (Cell *curr_cell : base_ptr_->part_ptr()->GetLocalCellPointers()) {
      auto &value_jumps_on_curr_cell = value_jumps.emplace_back();
      for (int i_node = 0; i_node < Cell::N; ++i_node) {
        auto &value_jump_on_curr_node = value_jumps_on_curr_cell[i_node];
        // Initialize value jumps to 0's:
        value_jump_on_curr_node.fill(0);
        // Update value jumps by each neighbor:
        Global const &global_i = curr_cell->integrator().GetGlobal(i_node);
        Value value_i = curr_cell->polynomial().GetValue(i_node);
        for (Cell *neighbor_i : curr_cell->adj_cells_) {
          Value jump_i = value_i - neighbor_i->polynomial().Extrapolate(global_i);
          mini::algebra::Maximize(&value_jump_on_curr_node, std::abs(jump_i));
        }
      }
    }
    assert(value_jumps.size() == part().CountLocalCells());
    return value_jumps;
  }

  std::vector<Value>
  IntegrateJumps(std::vector<std::array<Value, Cell::N>> const jumps) const {
    std::vector<Value> jump_integrals;
    jump_integrals.reserve(part().CountLocalCells());
    for (Cell *curr_cell : base_ptr_->part_ptr()->GetLocalCellPointers()) {
      auto &integral_on_curr_cell = jump_integrals.emplace_back(Value::Zero());
      auto &jump_on_curr_cell = jumps.at(curr_cell->id());
      auto const &integrator = curr_cell->integrator();
      assert(integrator.CountPoints() == Cell::N);
      for (int i_node = 0; i_node < Cell::N; ++i_node) {
        integral_on_curr_cell += integrator.GetGlobalWeight(i_node)
            * std::pow(jump_on_curr_cell[i_node], 2);
      }
    }
    assert(jump_integrals.size() == part().CountLocalCells());
    return jump_integrals;
  }

  std::vector<Value> GetViscosityValues(
      std::vector<Value> const &jump_integrals,
      std::vector<DampingMatrix> const &damping_matrices) const {
    std::vector<Value> viscosity_values;
    viscosity_values.reserve(part().CountLocalCells());
    for (Cell *curr_cell : base_ptr_->part_ptr()->GetLocalCellPointers()) {
      auto &viscosity_on_curr_cell = viscosity_values.emplace_back();
      auto &jump_integral_on_curr_cell = jump_integrals.at(curr_cell->id());
      auto &damping_matrix_on_curr_cell = damping_matrices.at(curr_cell->id());
      auto const &coeff = curr_cell->polynomial().coeff();
      assert(coeff.rows() == Cell::K);
      assert(coeff.cols() == Cell::N);
      for (int k = 0; k < Cell::K; ++k) {
        auto const &u_row = coeff.row(k);
        auto const &u_col = u_row.transpose();
        Scalar damping_rate = u_row.dot(damping_matrix_on_curr_cell * u_col);
        viscosity_on_curr_cell[k] = jump_integral_on_curr_cell[k]
            / (damping_rate * this->TimeScale());
      }
    }
    assert(viscosity_values.size() == part().CountLocalCells());
    return viscosity_values;
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
  Scalar TimeScale() const {
    return time_scale_;
  }
  Scalar &TimeScale() {
    return time_scale_;
  }

 private:
  Scalar time_scale_;

 protected:  // override virtual methods defined in Base
  void AddFluxDivergence(CellToFlux cell_to_flux, Cell const &cell,
      Scalar *data) const override {
    base().AddFluxDivergence(cell_to_flux, cell, data);
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
