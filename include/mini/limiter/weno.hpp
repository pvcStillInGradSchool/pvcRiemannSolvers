//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_LIMITER_WENO_HPP_
#define MINI_LIMITER_WENO_HPP_

#include <cassert>
#include <cmath>

#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "mini/basis/taylor.hpp"
#include "mini/polynomial/concept.hpp"
#include "mini/riemann/concept.hpp"

namespace mini {
namespace limiter {
namespace weno {

template <std::floating_point Scalar, int kComponents, int kDegrees>
class Smoothness {
  static constexpr int X{1}, Y{2}, Z{3};
  static constexpr int XX{4}, XY{5}, XZ{6}, YY{7}, YZ{8}, ZZ{9};
  static constexpr int XXX{10}, XXY{11}, XXZ{12}, XYY{13}, XYZ{14}, XZZ{15};
  static constexpr int YYY{16}, YYZ{17}, YZZ{18}, ZZZ{19};

 public:
  static constexpr int K = kComponents;
  static constexpr int N = basis::Taylor<Scalar, 3, kDegrees>::N;
  using MatKx1 = algebra::Matrix<Scalar, K, 1>;
  using MatKxN = algebra::Matrix<Scalar, K, N>;

  static MatKx1 GetSmoothness(const MatKxN &integral, Scalar volume)
      requires(kDegrees == 0) {
    MatKx1 smoothness; smoothness.setZero();
    return smoothness;
  }

  static MatKx1 GetSmoothness(const MatKxN &integral, Scalar volume)
      requires(kDegrees == 1) {
    MatKx1 smoothness = integral.col(X);
    smoothness += integral.col(Y);
    smoothness += integral.col(Z);
    return smoothness;
  }

  static MatKx1 GetSmoothness(const MatKxN &integral, Scalar volume)
      requires(kDegrees == 2) {
    auto w1  // weight of 1st-order partial derivatives
        = std::pow(volume, 2./3-1);
    MatKx1 smoothness = integral.col(X);
    smoothness += integral.col(Y);
    smoothness += integral.col(Z);
    smoothness *= w1;
    auto w2  // weight of 2nd-order partial derivatives
        = std::pow(volume, 4./3-1);
    smoothness += integral.col(XX) * w2;
    smoothness += integral.col(XY) * w2;
    smoothness += integral.col(XZ) * w2;
    smoothness += integral.col(YY) * w2;
    smoothness += integral.col(YZ) * w2;
    smoothness += integral.col(ZZ) * w2;
    return smoothness;
  }
  static MatKx1 GetSmoothness(const MatKxN &integral, Scalar volume)
      requires(kDegrees == 3) {
    auto w1  // weight of 1st-order partial derivatives
        = std::pow(volume, 2./3-1);
    MatKx1 smoothness = integral.col(X);
    smoothness += integral.col(Y);
    smoothness += integral.col(Z);
    smoothness *= w1;
    auto w2  // weight of 2nd-order partial derivatives
        = std::pow(volume, 4./3-1);
    smoothness += integral.col(XX) * w2;
    smoothness += integral.col(XY) * w2;
    smoothness += integral.col(XZ) * w2;
    smoothness += integral.col(YY) * w2;
    smoothness += integral.col(YZ) * w2;
    smoothness += integral.col(ZZ) * w2;
    auto w3  // weight of 3rd-order partial derivatives
        = volume;  // = std::pow(volume, 6./3-1);
    smoothness += integral.col(XXX) * w3;
    smoothness += integral.col(XXY) * w3;
    smoothness += integral.col(XXZ) * w3;
    smoothness += integral.col(XYY) * w3;
    smoothness += integral.col(XYZ) * w3;
    smoothness += integral.col(XZZ) * w3;
    smoothness += integral.col(YYY) * w3;
    smoothness += integral.col(YYZ) * w3;
    smoothness += integral.col(YZZ) * w3;
    smoothness += integral.col(ZZZ) * w3;
    return smoothness;
  }
};

template <mini::polynomial::Modal Projection>
auto GetSmoothness(const Projection &proj) {
  using Coeff = typename Projection::Coeff;
  using Global = typename Projection::Global;
  using Taylor = typename Projection::Taylor;
  using Scalar = typename Projection::Scalar;
  constexpr int K = Projection::K;
  constexpr int P = Projection::P;
  auto mat_pdv_func = [&proj](Global const &xyz) {
    auto local = xyz; local -= proj.center();
    auto mat_pdv = Taylor::GetPdvValue(local, proj.GetCoeffOnTaylorBasis());
    mat_pdv = mat_pdv.cwiseProduct(mat_pdv);
    return mat_pdv;
  };
  auto integral = integrator::Integrate(mat_pdv_func, proj.integrator());
  auto volume = proj.basis().Measure();
  return Smoothness<Scalar, K, P>::GetSmoothness(integral, volume);
}

template <typename Cell>
    requires mini::polynomial::Modal<typename Cell::Polynomial>
class Lazy {
 public:
  using Scalar = typename Cell::Scalar;
  using Polynomial = typename Cell::Polynomial;
  using ProjectionWrapper = typename Polynomial::Wrapper;
  using Basis = typename ProjectionWrapper::Basis;
  using Global = typename ProjectionWrapper::Global;
  using Value = typename ProjectionWrapper::Value;

 private:
  std::vector<ProjectionWrapper> old_projections_;
  ProjectionWrapper *new_projection_ptr_ = nullptr;
  const Cell *my_cell_ = nullptr;
  Value weights_;
  Scalar eps_;
  bool verbose_;

 public:
  Lazy(Scalar w0, Scalar eps, bool verbose = false)
      : eps_(eps), verbose_(verbose) {
    weights_.setOnes();
    weights_ *= w0;
  }
  bool IsNotSmooth(const Cell &cell) {
    return true;
  }
  ProjectionWrapper operator()(const Cell &cell) {
    my_cell_ = &cell;
    Borrow();
    Reconstruct();
    assert(new_projection_ptr_);
    return *new_projection_ptr_;
  }

 private:
   /**
   * @brief Borrow projections from adjacent cells.
   * 
   */
  void Borrow() {
    old_projections_.clear();
    old_projections_.reserve(my_cell_->adj_cells_.size() + 1);
    auto const &my_projection = my_cell_->polynomial().projection();
    auto my_average = my_projection.average();
    for (auto *adj_cell : my_cell_->adj_cells_) {
      assert(adj_cell);
      auto &adj_proj = old_projections_.emplace_back(my_projection.basis());
      assert(&(adj_proj.basis()) == &(my_projection.basis()));
      auto adj_func = [&adj_cell](Global const &global) {
        return adj_cell->polynomial().GlobalToValue(global);
      };
      adj_proj.Approximate(adj_func);
      adj_proj += my_average - adj_proj.average();
      if (verbose_) {
        std::cout << "\n  adj smoothness[" << adj_cell->metis_id << "] = ";
        std::cout << std::scientific << std::setprecision(3) <<
            GetSmoothness(adj_proj).transpose();
      }
    }
    old_projections_.emplace_back(my_projection);
    if (verbose_) {
      std::cout << "\n  old smoothness[" << my_cell_->metis_id << "] = ";
      std::cout << std::scientific << std::setprecision(3) <<
          GetSmoothness(old_projections_.back()).transpose();
    }
    new_projection_ptr_ = &(old_projections_.back());
    assert(&(new_projection_ptr_->basis()) == &(my_projection.basis()));
  }
  void Reconstruct() {
    int adj_cnt = my_cell_->adj_cells_.size();
    // initialize weights
    auto weights = std::vector<Value>(adj_cnt + 1, weights_);
    weights.back() *= -adj_cnt;
    weights.back().array() += 1.0;
    // modify weights by smoothness
    for (int i = 0; i <= adj_cnt; ++i) {
      auto beta = GetSmoothness(old_projections_[i]);
      beta.array() += eps_;
      beta.array() *= beta.array();
      weights[i].array() /= beta.array();
    }
    // normalize these weights
    Value sum; sum.setZero();
    sum = std::accumulate(weights.begin(), weights.end(), sum);
    assert(weights.size() == adj_cnt + 1);
    for (auto &weight : weights) {
      weight.array() /= sum.array();
    }
    // build the new (weighted) projection
    auto &new_projection = old_projections_.back();
    new_projection *= weights.back();
    for (int i = 0; i < adj_cnt; ++i) {
      old_projections_[i] *= weights[i];
      new_projection += old_projections_[i];
    }
  }
};

template <typename Cell, mini::riemann::Convective Riemann>
    requires mini::polynomial::Modal<typename Cell::Polynomial>
class Eigen {
 public:
  using Face = typename Cell::Face;
  using Scalar = typename Cell::Scalar;
  using Polynomial = typename Cell::Polynomial;
  using ProjectionWrapper = typename Polynomial::Wrapper;
  using Basis = typename ProjectionWrapper::Basis;
  using Global = typename ProjectionWrapper::Global;
  using Value = typename ProjectionWrapper::Value;

 private:
  std::function<std::vector<Riemann> const &(Face const &)> face_to_riemanns_;
  ProjectionWrapper new_projection_;
  std::vector<ProjectionWrapper> old_projections_;
  const Cell *my_cell_ = nullptr;
  Value weights_;
  Scalar eps_;
  Scalar total_volume_;

 public:
  template <typename FaceToRiemann>
  Eigen(Scalar w0, Scalar eps, FaceToRiemann &&face_to_riemanns)
      : eps_(eps), face_to_riemanns_(face_to_riemanns) {
    weights_.setOnes();
    weights_ *= w0;
  }
  static bool IsNotSmooth(const Cell &cell) {
    constexpr int components[] = { 0, Cell::K-1 };
    auto max_abs_averages = cell.polynomial().average();
    for (int i : components) {
      max_abs_averages[i] = std::max(1e-9, std::abs(max_abs_averages[i]));
    }
    typename Cell::Value sum_abs_differences; sum_abs_differences.setZero();
    auto my_values = cell.GlobalToValue(cell.center());
    for (const Cell *adj_cell : cell.adj_cells_) {
      auto adj_values = adj_cell->GlobalToValue(cell.center());
      auto adj_averages = adj_cell->polynomial().average();
      for (int i : components) {
        sum_abs_differences[i] += std::abs(my_values[i] - adj_values[i]);
        max_abs_averages[i] = std::max(max_abs_averages[i],
            std::abs(adj_averages[i]));
      }
    }
    constexpr auto volume_power = (Cell::P + 1.0) / 2.0 / Cell::D;
    auto divisor = std::pow(cell.volume(), volume_power);
    divisor *= cell.adj_cells_.size();
    constexpr auto smoothness_reference = Cell::P < 3 ? 1.0 : 3.0;
    for (int i : components) {
      auto smoothness = sum_abs_differences[i] / max_abs_averages[i] / divisor;
      if (smoothness > smoothness_reference) {
        return true;  // if any component is not smooth
      }
    }
    return false;  // if all components are smooth
  }
  ProjectionWrapper operator()(const Cell &cell) {
    my_cell_ = &cell;
    Borrow();
    Reconstruct();
    return new_projection_;
  }

 private:
  /**
   * @brief Borrow projections from adjacent cells.
   * 
   */
  void Borrow() {
    old_projections_.clear();
    old_projections_.reserve(my_cell_->adj_cells_.size() + 1);
    auto const &my_projection = my_cell_->polynomial().projection();
    auto my_average = my_projection.average();
    for (auto *adj_cell : my_cell_->adj_cells_) {
      auto &adj_proj = old_projections_.emplace_back(my_projection.basis());
      auto adj_func = [&adj_cell](Global const &global) {
        return adj_cell->polynomial().GlobalToValue(global);
      };
      adj_proj.Approximate(adj_func);
      adj_proj += my_average - adj_proj.average();
    }
    old_projections_.emplace_back(my_projection);
  }
  /**
   * @brief Rotate borrowed projections onto the interface between cells
   * 
   */
  void ReconstructOnFace(const Face &adj_face) {
    assert(my_cell_->adj_faces_.size() == my_cell_->adj_cells_.size());
    int adj_cnt = my_cell_->adj_faces_.size();
    // build eigen-matrices in the rotated coordinate system
    const auto &big_u = my_cell_->polynomial().average();
    // TODO(PVC): take average of eigen matrices on all quadrature points
    const auto &const_riemann = face_to_riemanns_(adj_face).at(0);
    auto *riemann = const_cast<Riemann *>(&const_riemann);
    riemann->UpdateEigenMatrices(big_u);
    // initialize weights
    auto weights = std::vector<Value>(adj_cnt + 1, weights_);
    weights.back() *= -adj_cnt;
    weights.back().array() += 1.0;
    // modify weights by smoothness
    auto rotated_projections = old_projections_;
    for (int i = 0; i <= adj_cnt; ++i) {
      auto &projection_i = rotated_projections[i];
      projection_i.LeftMultiply(riemann->L());
      auto beta = GetSmoothness(projection_i);
      beta.array() += eps_;
      beta.array() *= beta.array();
      weights[i].array() /= beta.array();
    }
    // normalize these weights
    Value sum; sum.setZero();
    sum = std::accumulate(weights.begin(), weights.end(), sum);
    assert(weights.size() == adj_cnt + 1);
    for (auto &weight : weights) {
      weight.array() /= sum.array();
    }
    // build the new (weighted) projection
    auto &new_projection = rotated_projections.back();
    new_projection *= weights.back();
    for (int i = 0; i < adj_cnt; ++i) {
      rotated_projections[i] *= weights[i];
      new_projection += rotated_projections[i];
    }
    // rotate the new projection back to the global system
    new_projection.LeftMultiply(riemann->R());
    // scale the new projection by volume
    auto adj_volume = adj_face.other(my_cell_)->volume();
    new_projection *= adj_volume;
    new_projection_ += new_projection;
    total_volume_ += adj_volume;
  }
  /**
   * @brief Reconstruct projections by weights
   * 
   */
  void Reconstruct() {
    auto const &my_basis = my_cell_->polynomial().projection().basis();
    new_projection_ = ProjectionWrapper(my_basis);
    new_projection_.coeff().setZero();
    total_volume_ = 0.0;
    for (auto *adj_face : my_cell_->adj_faces_) {
      ReconstructOnFace(*adj_face);
    }
    new_projection_ /= total_volume_;
  }
};

template <typename Cell>
    requires mini::polynomial::Modal<typename Cell::Polynomial>
class Dummy {
 public:
  using Scalar = typename Cell::Scalar;
  using Polynomial = typename Cell::Polynomial;
  using ProjectionWrapper = typename Polynomial::Wrapper;

 public:
  Dummy(Scalar w0, Scalar eps, bool verbose = false) {
  }
  bool IsNotSmooth(const Cell &cell) {
    return true;
  }
  ProjectionWrapper operator()(const Cell &cell) {
    return ProjectionWrapper(cell.polynomial());
  }
};

}  // namespace weno
}  // namespace limiter
}  // namespace mini

#endif  // MINI_LIMITER_WENO_HPP_
