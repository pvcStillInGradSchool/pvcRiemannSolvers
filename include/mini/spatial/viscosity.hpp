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
  using Base = R;

  // members required by mini::riemann::Convective
  static constexpr int kComponents = Base::kComponents;
  static constexpr int kDimensions = Base::kDimensions;
  using Convection = typename Base::Convection;
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using Conservative = typename Base::Conservative;
  using Flux = typename Base::Flux;
  using FluxMatrix = typename Base::FluxMatrix;
  using Jacobian = typename Base::Jacobian;

  // members required by mini::riemann::Diffusion
  using Diffusion = EnergyBasedViscosity;
  using Gradient = mini::algebra::Matrix<Scalar, kDimensions, kComponents>;
  using Property = mini::algebra::Vector<Scalar, kComponents>;
  static std::vector<std::vector<Property>> properties_;  // [i_cell][i_node]

  // members derived from Part
  using Part = P;
  using Face = typename Part::Face;
  using Cell = typename Part::Cell;
  using Global = typename Cell::Global;
  using Local = typename Cell::Local;
  using Polynomial = typename Cell::Polynomial;
  using Coeff = typename Polynomial::Coeff;
  using Value = typename Polynomial::Value;
  static_assert(std::is_same_v<Value, Property>);

  // override methods in Base::Diffusion
  template <typename Int>
  static Property const &GetPropertyOnCell(Int i_cell, int i_node) {
    return properties_.at(i_cell).at(i_node);
  }

  static void MinusViscousFlux(FluxMatrix *flux, Property const &nu,
      Conservative const &value, Gradient const &gradient) {
    for (int k = 0; k < kComponents; ++k) {
      flux->row(k) -= nu[k] * gradient.col(k);
    }
  }

  static void MinusViscousFlux(Flux *flux, Property const &nu,
      Conservative const &value, Gradient const &gradient,
      Vector const &normal) {
    for (int k = 0; k < kComponents; ++k) {
      (*flux)[k] -= nu[k] * normal.dot(gradient.col(k));
    }
  }

  static void MinusViscousFluxOnNoSlipWall(Flux *flux,
      Property const &nu, Value const &wall_value,
      Conservative const &c_val, Gradient const &c_grad,
      Vector const &normal, Scalar value_penalty) {
    MinusViscousFlux(flux, nu, c_val, c_grad, normal);
  }

  /**
   * @brief non-constant condition for static assertion 
   * 
   */
  // static_assert(mini::riemann::Convective<EnergyBasedViscosity>);
  using Spatial = FiniteElement<Part, EnergyBasedViscosity>;

 private:
  static std::vector<std::vector<Scalar>> send_bufs_, recv_bufs_;
  static std::vector<MPI_Request> requests_;

 public:
  /**
   * @brief Initialize data structures for sharing viscosity properties across `Part`s.
   * 
   * It should be called once and only once before the main loop.
   */
  static void InitializeRequestsAndBuffers() {
    part().InitializeRequestsAndBuffers(Cell::kFields,
        &requests_, &send_bufs_, &recv_bufs_);
  }

 private:
  static Spatial *spatial_ptr_;
  static Scalar time_scale_;

 public:
  static void InstallSpatial(Spatial *spatial_ptr) {
    spatial_ptr_ = spatial_ptr;
    properties_.resize(part().CountLocalCells() + part().CountGhostCells());
    for (Cell *cell_ptr: part_ptr()->GetLocalCellPointers()) {
      auto n_point = cell_ptr->integrator().CountPoints();
      properties_.at(cell_ptr->id()).resize(n_point);
    }
    for (Cell const &cell: part().GetGhostCells()) {
      auto n_point = cell.integrator().CountPoints();
      properties_.at(cell.id()).resize(n_point);
    }
  }

  static Part *part_ptr() {
    return spatial_ptr_->part_ptr();
  }
  static Part const &part() {
    return spatial_ptr_->part();
  }

  static Scalar GetTimeScale() {
    return time_scale_;
  }
  static void SetTimeScale(Scalar time_scale) {
    time_scale_ = time_scale;
  }

 public:
  using Base::Base;
  EnergyBasedViscosity(const EnergyBasedViscosity &) = default;
  EnergyBasedViscosity &operator=(const EnergyBasedViscosity &) = default;
  EnergyBasedViscosity(EnergyBasedViscosity &&) noexcept = default;
  EnergyBasedViscosity &operator=(EnergyBasedViscosity &&) noexcept = default;
  ~EnergyBasedViscosity() noexcept = default;

 private:  // methods used in BuildDampingMatrices()
  /**
   * @brief Set the viscous Property of a given Cell and all it's neighbors to a given value.
   * 
   */
  static void SetViscousProperty(Cell *curr_cell, Property const &property_given) {
    auto SetProperty = [&property_given](Cell *cell_ptr) {
      for (auto &property : properties_.at(cell_ptr->id())) {
        property = property_given;
      }
    };
    SetProperty(curr_cell);
    for (Cell *neighobor : curr_cell->adj_cells_) {
      SetProperty(neighobor);
    }
  }
  static void UpdateCellResidual(Cell *curr_cell, Scalar *residual_data) {
    spatial_ptr_->AddFluxDivergence(*curr_cell, residual_data);
    Coeff dummy;
    for (Face *face : curr_cell->adj_faces_) {
      if (face->holder_ptr() == curr_cell) {
        spatial_ptr_->AddFluxToHolder(*face, residual_data);
        // spatial_ptr_->AddFluxToHolderAndSharer(*face, residual_data, nullptr);
      } else {
        spatial_ptr_->AddFluxToSharer(*face, residual_data);
        // spatial_ptr_->AddFluxToHolderAndSharer(*face, dummy.data(), residual_data);
      }
#if !defined(NDEBUG) && defined(ENABLE_SLOW_CONSISTENCY_CHECK)
      Cell const *other = face->other(curr_cell);
      assert(other->polynomial().coeff() == Coeff::Zero());
      Coeff res1, res2;
      if (face->holder_ptr() == curr_cell) {
        res1.setZero();
        res2.setZero();
        spatial_ptr_->AddFluxToHolder(*face, res1.data());
        spatial_ptr_->AddFluxToHolderAndSharer(*face, res2.data(), nullptr);
        assert(res1 == res2);
      } else {
        assert(face->holder_ptr() == other);
        res1.setZero();
        res2.setZero();
        spatial_ptr_->AddFluxToSharer(*face, res1.data());
        spatial_ptr_->AddFluxToHolderAndSharer(*face, dummy.data(), res2.data());
        assert(res1 == res2);
      }
#endif
    }
    assert(curr_cell->adj_cells_.size() == curr_cell->adj_faces_.size());
    for (Face *face : curr_cell->boundary_faces_) {
      assert(face->holder_ptr() == curr_cell);
      assert(nullptr == face->other(curr_cell));
      assert(face->HolderToSharer().dot(
          face->center() - curr_cell->center()) > 0);
      spatial_ptr_->AddFluxToHolder(*face, residual_data);
    }
    assert(curr_cell->boundary_faces_.size() + curr_cell->adj_faces_.size() == 6);
  }

 public:  // methods for generating artificial viscosity
  using DampingMatrix = algebra::Matrix<Scalar, Cell::N, Cell::N>;

  static std::vector<DampingMatrix> BuildDampingMatrices() {
#if !defined(NDEBUG) && defined(ENABLE_SLOW_CONSISTENCY_CHECK)
    std::srand(31415926);
#endif
    auto matrices = std::vector<DampingMatrix>(part().CountLocalCells());
    for (Cell *curr_cell: part_ptr()->GetLocalCellPointers()) {
      // Nullify coeffs and properties on all its neighbors:
      for (Cell *neighbor : curr_cell->adj_cells_) {
        neighbor->polynomial().SetZero();
      }
      // Build the damping matrix column by column:
      auto &matrix = matrices.at(curr_cell->id());
      auto GetCellResidual = [](Cell *cell_ptr) -> Coeff {
        Coeff residual; residual.setZero();
        SetViscousProperty(cell_ptr, Value::Zero());
        UpdateCellResidual(cell_ptr, residual.data());
        residual = -residual;
        SetViscousProperty(cell_ptr, Value::Ones());
        UpdateCellResidual(cell_ptr, residual.data());
        return residual;
      };
      auto &solution = curr_cell->polynomial().coeff();
      solution.setZero();
      for (int c = 0; c < Cell::N; ++c) {
        solution.col(c).setOnes();
        if (c > 0) {
          solution.col(c - 1).setZero();
        }
        Coeff residual = GetCellResidual(curr_cell);
        // Write the residual column into the matrix:
        matrix.col(c) = residual.row(0);
        for (int r = 1; r < Cell::K; ++r) {
          assert((residual.row(r) - residual.row(0)).squaredNorm() < 1e-10);
        }
      }
#if !defined(NDEBUG) && defined(ENABLE_SLOW_CONSISTENCY_CHECK)
      solution = Coeff::Random();
      Coeff residual = GetCellResidual(curr_cell);
      for (int k = 0; k < Cell::K; ++k) {
        auto const &residual_col = residual.row(k).transpose();
        auto const &solution_col = solution.row(k).transpose();
        assert((residual_col - matrix * solution_col).norm() < 1e-10);
      }
#endif
      // Scale the damping matrix by local weight over Jacobian:
      assert(curr_cell->polynomial().kLocal);
      for (int r = 0; r < Cell::N; ++r) {
        Scalar scale
            = curr_cell->integrator().GetLocalWeight(r)
            / curr_cell->integrator().GetJacobianDeterminant(r);
        assert(scale > 0);
        matrix.row(r) *= scale;
      }
    }
    return matrices;
  }

  static std::vector<std::array<Value, Cell::N>> BuildValueJumps() {
    std::vector<std::array<Value, Cell::N>> value_jumps;
    value_jumps.reserve(part().CountLocalCells());
    for (Cell *curr_cell : part_ptr()->GetLocalCellPointers()) {
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
    for (Cell *curr_cell : part_ptr()->GetLocalCellPointers()) {
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
    for (Cell *curr_cell : part_ptr()->GetLocalCellPointers()) {
      auto &viscosity_on_curr_cell = viscosity_values.emplace_back();
      auto &jump_integral_on_curr_cell = jump_integrals.at(curr_cell->id());
      auto &damping_matrix_on_curr_cell = damping_matrices.at(curr_cell->id());
      auto const &coeff = curr_cell->polynomial().coeff();
      assert(coeff.rows() == Cell::K);
      assert(coeff.cols() == Cell::N);
      for (int k = 0; k < Cell::K; ++k) {
        auto const &u_row = coeff.row(k);
        auto const &u_col = u_row.transpose();
        Scalar damping_rate = -u_row.dot(damping_matrix_on_curr_cell * u_col);
        viscosity_on_curr_cell[k] = jump_integral_on_curr_cell[k]
            / (damping_rate * GetTimeScale());
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
std::vector<std::vector<typename EnergyBasedViscosity<P, R>::Property>>
EnergyBasedViscosity<P, R>::properties_;

template <typename P, typename R>
std::vector<std::vector<typename EnergyBasedViscosity<P, R>::Scalar>>
EnergyBasedViscosity<P, R>::send_bufs_;

template <typename P, typename R>
std::vector<std::vector<typename EnergyBasedViscosity<P, R>::Scalar>>
EnergyBasedViscosity<P, R>::recv_bufs_;

template <typename P, typename R>
std::vector<MPI_Request>
EnergyBasedViscosity<P, R>::requests_;

template <typename P, typename R>
typename EnergyBasedViscosity<P, R>::Spatial *
EnergyBasedViscosity<P, R>::spatial_ptr_;

template <typename P, typename R>
typename EnergyBasedViscosity<P, R>::Scalar
EnergyBasedViscosity<P, R>::time_scale_;

}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_VISCOSITY_HPP_
