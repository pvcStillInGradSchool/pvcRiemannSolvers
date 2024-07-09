// Copyright 2024 PEI Weicheng
#ifndef MINI_SPATIAL_VISCOSITY_HPP_
#define MINI_SPATIAL_VISCOSITY_HPP_

#include <concepts>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#ifndef NDEBUG
#include <fstream>
#include <iomanip>
#endif

#include "mini/algebra/eigen.hpp"
#include "mini/spatial/fem.hpp"
#include "mini/riemann/concept.hpp"
#include "mini/riemann/diffusive/linear.hpp"
#include "mini/riemann/diffusive/direct.hpp"

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

  void MinusViscousFluxOnNeumannWall(Flux *flux, Property const &nu,
      Conservative const &c_val) const {
    typename Convection::Conservative value_jump =
        this->Convection::MinusMirroredValue(c_val);
    Scalar penalty = this->Diffusion::GetValuePenalty();
    flux->array() -= nu.array() * value_jump.array() * penalty;
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
    part().InitializeRequestsAndBuffers(kComponents,
        &requests_, &send_bufs_, &recv_bufs_);
  }

  static void ShareGhostCellProperties() {
    auto operation = [](Cell const *cell_ptr, Scalar *buf) -> Scalar * {
      std::vector<Property> const &properties = properties_.at(cell_ptr->id());
      assert(Cell::N == properties.size());
      static_assert(Cell::K * sizeof(Scalar) == sizeof(Property));
      // Since properties[0] == ... == properties[N-1], only one has to be sent.
      std::memcpy(buf, properties.data(), sizeof(Property));
      return buf + Cell::K;
    };
    part_ptr()->ShareGhostCellData(&requests_, &send_bufs_, &recv_bufs_,
        operation);
  }

  static void UpdateGhostCellProperties() {
    auto operation = [](Cell *cell_ptr, Scalar const *buf) -> Scalar const * {
      std::vector<Property> &properties = properties_.at(cell_ptr->id());
      assert(Cell::N == properties.size());
      static_assert(Cell::K * sizeof(Scalar) == sizeof(Property));
      std::ranges::fill(properties, *reinterpret_cast<Property const *>(buf));
      return buf + Cell::K;
    };
    part_ptr()->UpdateGhostCellData(&requests_, &recv_bufs_, operation);
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
    assert(spatial_ptr_);
    return spatial_ptr_->part_ptr();
  }
  static Part const &part() {
    return *part_ptr();
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
      std::ranges::fill(properties_.at(cell_ptr->id()), property_given);
    };
    SetProperty(curr_cell);
    for (Cell *neighobor : curr_cell->adj_cells_) {
      SetProperty(neighobor);
    }
  }
  static void UpdateCellResidual(Cell *curr_cell, Scalar *residual_data) {
    spatial_ptr_->AddFluxDivergence(*curr_cell, residual_data);
    for (Face *face : curr_cell->adj_faces_) {
      if (face->holder_ptr() == curr_cell) {
        spatial_ptr_->AddFluxToHolder(*face, residual_data);
      } else {
        spatial_ptr_->AddFluxToSharer(*face, residual_data);
      }
#if !defined(NDEBUG) && defined(ENABLE_SLOW_CONSISTENCY_CHECK)
{
      Coeff dummy;
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

 private:
  static std::vector<DampingMatrix> damping_matrices_;

 public:
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
        if (!Polynomial::kLocal) {
          // TODO(PVC): Use the virtual method in Base
          Scalar *data = residual.data();
          const auto &integrator = cell_ptr->integrator();
          for (int q = 0; q < Cell::N; ++q) {
            auto scale = 1. / integrator.GetJacobianDeterminant(q);
            data = cell_ptr->polynomial().ScaleValueAt(scale, data);
          }
          assert(data == residual.data() + Cell::kFields);
        }
        return residual;
      };
      auto &curr_polynomial = curr_cell->polynomial();
      curr_polynomial.SetZero();
      for (int c = 0; c < Cell::N; ++c) {
        if (c > 0) {
          curr_polynomial.SetCoeff(c - 1, Value::Zero());
        }
        curr_polynomial.SetCoeff(c, Value::Ones());
        Coeff residual = GetCellResidual(curr_cell);
        // Write the residual column into the matrix:
        matrix.col(c) = residual.row(0);
#ifndef NDEBUG
        for (int r = 1; r < Cell::K; ++r) {
          if ((residual.row(r) - residual.row(0)).squaredNorm() > 1e-10) {
            std::cout << residual.row(r) << "\n\n";
            std::cout << residual.row(0) << "\n\n";
            assert(false);
          }
        }
#endif
      }
#if !defined(NDEBUG) && defined(ENABLE_SLOW_CONSISTENCY_CHECK)
{
      Coeff solution = Coeff::Random();
      curr_cell->polynomial().SetCoeff(solution);
      assert(solution == curr_cell->polynomial().coeff());
      Coeff residual = GetCellResidual(curr_cell);
      assert(solution == curr_cell->polynomial().coeff());
      for (int k = 0; k < Cell::K; ++k) {
        auto const &residual_col = residual.row(k).transpose();
        auto const &solution_col = solution.row(k).transpose();
        if ((residual_col - matrix * solution_col).norm() > 1e-9) {
          std::cout << (residual_col).transpose() << "\n\n";
          std::cout << (matrix * solution_col).transpose() << "\n\n";
          std::cout << (residual_col - matrix * solution_col).norm() << "\n\n";
          assert(false);
        }
      }
}
#endif
      // Scale the damping matrix by Gaussian weights:
      for (int r = 0; r < Cell::N; ++r) {
        Scalar scale = curr_cell->integrator().GetLocalWeight(r);
        Scalar det = curr_cell->integrator().GetJacobianDeterminant(r);
        if (Polynomial::kLocal) {
          scale /= det;
        } else {
          scale *= det;
          assert(scale == curr_cell->integrator().GetGlobalWeight(r));
        }
        assert(scale > 0);
        matrix.row(r) *= scale;
      }
    }
    return matrices;
  }

 private:  // methods for getting the square of jump on a given quadrature point
  static Value Maximize(Cell const &cell, int i_node) {
    Value jump_on_curr_node = Value::Zero();
    Global const &global = cell.integrator().GetGlobal(i_node);
    Value const &value = cell.polynomial().GetValue(i_node);
    for (Cell *neighbor : cell.adj_cells_) {
      Value jump = value - neighbor->polynomial().Extrapolate(global);
      mini::algebra::Maximize(&jump_on_curr_node, std::pow(jump, 2));
    }
    return jump_on_curr_node;
  }

  static Value WeightedAverage(Cell const &cell, int i_node) {
    Value jump_on_curr_node = Value::Zero();
    Global const &global = cell.integrator().GetGlobal(i_node);
    Value const &value = cell.polynomial().GetValue(i_node);
    Scalar weight_sum = 0;
    for (Cell *neighbor : cell.adj_cells_) {
      Value jump = value - neighbor->polynomial().Extrapolate(global);
      // the closer neighbor, the larger weight:
      Scalar weight = std::exp(-(neighbor->center() - global).squaredNorm());
      jump_on_curr_node += std::pow(jump, 2) * weight;
      weight_sum += weight;
    }
    return jump_on_curr_node /= weight_sum;
  }

 private:
  static std::vector<std::array<Value, Cell::N>> BuildValueJumps() {
    std::vector<std::array<Value, Cell::N>> value_jumps;
    value_jumps.reserve(part().CountLocalCells());
    for (Cell *curr_cell : part_ptr()->GetLocalCellPointers()) {
      auto &value_jumps_on_curr_cell = value_jumps.emplace_back();
      for (int i_node = 0; i_node < Cell::N; ++i_node) {
        value_jumps_on_curr_cell[i_node] = WeightedAverage(*curr_cell, i_node);
      }
    }
    assert(value_jumps.size() == part().CountLocalCells());
    return value_jumps;
  }

 public:
  static std::vector<Value> IntegrateJumpOnCells() {
    std::vector<Value> jump_integrals;
    jump_integrals.reserve(part().CountLocalCells());
    auto jumps = BuildValueJumps();
    for (Cell *curr_cell : part_ptr()->GetLocalCellPointers()) {
      auto &integral_on_curr_cell = jump_integrals.emplace_back(Value::Zero());
      auto &jump_on_curr_cell = jumps.at(curr_cell->id());
      auto const &integrator = curr_cell->integrator();
      assert(integrator.CountPoints() == Cell::N);
      for (int i_node = 0; i_node < Cell::N; ++i_node) {
        integral_on_curr_cell += integrator.GetGlobalWeight(i_node)
            * jump_on_curr_cell[i_node];
      }
    }
    assert(jump_integrals.size() == part().CountLocalCells());
    return jump_integrals;
  }

  static std::vector<Value> IntegrateJumpOnFaces() {
    std::vector<Value> jump_integrals;
    jump_integrals.resize(part().CountLocalCells(), Value::Zero());
    for (Face const &face : part_ptr()->GetLocalFaces()) {
      const auto &holder = face.holder();
      const auto &sharer = face.sharer();
      const auto &integrator = face.integrator();
      for (int i = 0; i < integrator.CountPoints(); ++i) {
        Value jump = spatial_ptr_->GetValueJump(face, i);
        jump.array() *= jump.array();
        jump_integrals.at(holder.id()) += jump
            * (integrator.GetGlobalWeight(i) * face.holder_height());
        jump_integrals.at(sharer.id()) += jump
            * (integrator.GetGlobalWeight(i) * face.sharer_height());
      }
    }
    for (Face const &face : part_ptr()->GetGhostFaces()) {
      const auto &holder = face.holder();
      const auto &sharer = face.sharer();
      const auto &integrator = face.integrator();
      for (int i = 0; i < integrator.CountPoints(); ++i) {
        Value jump = spatial_ptr_->GetValueJump(face, i);
        jump.array() *= jump.array();
        jump_integrals.at(holder.id()) += jump
            * (integrator.GetGlobalWeight(i) * face.holder_height());
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
      auto GetMaximumSpeedAndReferenceValueSquare = [](Cell const &cell)
            -> std::pair<Scalar, Value> {
        Scalar speed, max_speed{ 1e-5/* avoid 0 in divisor */ };
        typename Convection::Conservative squares,
            max_squares{ Value::Ones() * 1e-5/* avoid 0 in divisor */ };
        auto const &polynomial = cell.polynomial();
        for (int i = 0; i < Cell::N; ++i) {
            speed = Convection::GetMaximumSpeedAndReferenceValueSquare(
                polynomial.GetValue(i), &squares);
            max_speed = std::max(max_speed, speed);
            mini::algebra::Maximize(&max_squares, squares);
        }
        return { max_speed, max_squares };
      };
      auto [max_speed, reference_value_square]
          = GetMaximumSpeedAndReferenceValueSquare(*curr_cell);
      Scalar cell_length = curr_cell->length();
      Scalar max_viscosity = max_speed * cell_length / Cell::P;
      assert(!std::isinf(max_viscosity) && !std::isnan(max_viscosity));
      Scalar time_base = cell_length / max_speed;
      reference_value_square *= std::pow(cell_length,
          (Cell::P + 1)/* accuracy */ + 3/* volume */);
      // Now, reference_value_square[k] == |u_k|^2 * h_j^{P + 1} * |E_j|
      for (int k = 0; k < Cell::K; ++k) {
        if (jump_integral_on_curr_cell[k] < reference_value_square[k]) {
          viscosity_on_curr_cell[k] = 0.;
          continue;
        }
        auto const &u_row = coeff.row(k);
        auto const &u_col = u_row.transpose();
        Scalar damping_rate = -u_row.dot(damping_matrix_on_curr_cell * u_col);
        assert(damping_rate >= 0);
        Scalar damping_time = time_base * GetTimeScale();
        viscosity_on_curr_cell[k] = std::min(max_viscosity, std::max(0.0,
            // to protect against damping_rate == -0.0, which would lead to -inf
            jump_integral_on_curr_cell[k] / (damping_rate * damping_time)));
#ifndef NDEBUG
        if (std::isinf(viscosity_on_curr_cell[k])) {
          throw std::runtime_error("inf viscosity_on_curr_cell[k]: "
              + std::to_string(jump_integral_on_curr_cell[k]) + " / "
              + std::to_string(damping_rate * damping_time));
        }
#endif
      }
#ifndef NDEBUG
      std::fstream log{ "damping" + std::to_string(curr_cell->metis_id) + ".txt", log.out };
      log << std::scientific << std::setprecision(2) << damping_matrix_on_curr_cell << "\n";
#endif
    }
    assert(viscosity_values.size() == part().CountLocalCells());
    return viscosity_values;
  }

 public:  // wrappers to be used in Spatial
  using Viscosity = EnergyBasedViscosity;

  static void Initialize(Spatial *spatial_ptr) {
    InstallSpatial(spatial_ptr);
    InitializeRequestsAndBuffers();
    damping_matrices_ = BuildDampingMatrices();
    SetTimeScale(1.0);
  }

  static void UpdateProperties() {
    // auto jump_integrals = IntegrateJumpOnCells();
    auto jump_integrals = IntegrateJumpOnFaces();
    auto viscosity_values = GetViscosityValues(
        jump_integrals, damping_matrices_);
    assert(properties_.size()
        == part().CountGhostCells() + part().CountLocalCells());
    assert(viscosity_values.size() == part().CountLocalCells());
    for (ssize_t i_cell = 0; i_cell < part().CountLocalCells(); ++i_cell) {
      std::ranges::fill(properties_.at(i_cell), viscosity_values.at(i_cell));
    }
    /* TODO(PVC): replace by std::views::zip in C++23
    for (auto &[properties, viscosity_value]
        : std::views::zip(properties_, viscosity_values)) {
      std::ranges::fill(properties, viscosity_value);
     */
  }
};

template <typename P, typename R>
std::vector<typename EnergyBasedViscosity<P, R>::DampingMatrix>
EnergyBasedViscosity<P, R>::damping_matrices_;

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
