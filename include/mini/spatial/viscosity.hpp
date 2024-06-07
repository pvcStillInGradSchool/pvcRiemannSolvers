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

#include <fstream>
#include <iomanip>

#include "mini/algebra/eigen.hpp"
#include "mini/spatial/fem.hpp"
#include "mini/riemann/concept.hpp"
#include "mini/riemann/diffusive/linear.hpp"
#include "mini/riemann/diffusive/direct_dg.hpp"

namespace mini {
namespace spatial {

template <typename P, mini::riemann::Convective R>
class EnergyBasedViscosity : public R {
 public:
  using Spatial = FiniteElement<P, EnergyBasedViscosity>;
  using Part = typename Spatial::Part;
  using Riemann = typename Spatial::Riemann;
  using Scalar = typename Spatial::Scalar;
  using Face = typename Spatial::Face;
  using Cell = typename Spatial::Cell;
  using Global = typename Spatial::Global;
  using Polynomial = typename Spatial::Polynomial;
  using Integrator = typename Polynomial::Integrator;
  using Local = typename Integrator::Local;
  using Coeff = typename Spatial::Coeff;
  using Value = typename Spatial::Value;
  using Temporal = typename Spatial::Temporal;
  using Column = typename Spatial::Column;
  using FluxMatrix = typename Spatial::FluxMatrix;
  using CellToFlux = typename Spatial::CellToFlux;

 private:
  static Spatial *spatial_ptr_;
  static Scalar time_scale_;

 public:
  static void InstallSpatial(Spatial *spatial_ptr)
    spatial_ptr_ = spatial_ptr {
  }

  static Part *part_ptr() {
    return spatial_ptr_->part_ptr();
  }
  static Part const &part() {
    return spatial_ptr_->part();
  }

  static Scalar TimeScale() const {
    return time_scale_;
  }
  static Scalar &TimeScale() {
    return time_scale_;
  }

 private:
  // TODO(PVC): avoid overriding Spatial::Riemann
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
  EnergyBasedViscosity(const EnergyBasedViscosity &) = default;
  EnergyBasedViscosity &operator=(const EnergyBasedViscosity &) = default;
  EnergyBasedViscosity(EnergyBasedViscosity &&) noexcept = default;
  EnergyBasedViscosity &operator=(EnergyBasedViscosity &&) noexcept = default;
  ~EnergyBasedViscosity() noexcept = default;

 public:  // methods for generating artificial viscosity
  using DampingMatrix = algebra::Matrix<Scalar, Cell::N, Cell::N>;

  static std::vector<DampingMatrix> BuildDampingMatrices() {
    auto matrices = std::vector<DampingMatrix>(part().CountLocalCells());
    Diffusion::SetProperty(1.0);
    for (Cell *cell_ptr: spatial_ptr_->part_ptr()->GetLocalCellPointers()) {
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
        spatial_ptr_->AddFluxDivergence(GetDiffusiveFluxMatrix, *cell_ptr,
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
      spatial_ptr_->AddFluxDivergence(GetDiffusiveFluxMatrix, *cell_ptr,
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

  static std::vector<std::array<Value, Cell::N>> BuildValueJumps() {
    std::vector<std::array<Value, Cell::N>> value_jumps;
    value_jumps.reserve(part().CountLocalCells());
    for (Cell *curr_cell : spatial_ptr_->part_ptr()->GetLocalCellPointers()) {
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

  static std::vector<Value>
  IntegrateJumps(std::vector<std::array<Value, Cell::N>> const jumps) {
    std::vector<Value> jump_integrals;
    jump_integrals.reserve(part().CountLocalCells());
    for (Cell *curr_cell : spatial_ptr_->part_ptr()->GetLocalCellPointers()) {
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

  static std::vector<Value> GetViscosityValues(
      std::vector<Value> const &jump_integrals,
      std::vector<DampingMatrix> const &damping_matrices) {
    std::vector<Value> viscosity_values;
    viscosity_values.reserve(part().CountLocalCells());
    for (Cell *curr_cell : spatial_ptr_->part_ptr()->GetLocalCellPointers()) {
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
            / (damping_rate * TimeScale());
      }
      if (curr_cell->id() == 0) {
        std::fstream log{ "damping" + std::to_string(part().mpi_rank()) + ".txt", log.out };
        log << std::scientific << std::setprecision(2) << damping_matrix_on_curr_cell << "\n";
      }
    }
    assert(viscosity_values.size() == part().CountLocalCells());
    return viscosity_values;
  }
};

template <typename P, typename R>
typename EnergyBasedViscosity<P, R>::Spatial *
EnergyBasedViscosity<P, R>::spatial_ptr_;

template <typename P, typename R>
typename EnergyBasedViscosity<P, R>::Scalar
EnergyBasedViscosity<P, R>::time_scale_;

}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_VISCOSITY_HPP_
