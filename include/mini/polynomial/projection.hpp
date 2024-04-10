//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_POLYNOMIAL_PROJECTION_HPP_
#define MINI_POLYNOMIAL_PROJECTION_HPP_

#include <concepts>

#include <cmath>
#include <cstring>

#include <iostream>
#include <type_traits>
#include <utility>

#include "mini/algebra/eigen.hpp"
#include "mini/gauss/legendre.hpp"
#include "mini/gauss/hexahedron.hpp"
#include "mini/gauss/function.hpp"
#include "mini/basis/linear.hpp"

namespace mini {
namespace polynomial {

namespace projection {

template <typename Projection, typename Callable>
void Project(Projection *proj, Callable &&func) {
  using Global = typename Projection::Global;
  using Mat1xN = typename Projection::Mat1xN;
  using Value = typename Projection::Value;
  using Coeff = typename Projection::Coeff;
  using Return = std::invoke_result_t<Callable, Global>;
  static_assert(std::is_same_v<Return, Value> || std::is_scalar_v<Return>);
  proj->coeff() = gauss::Integrate([&](Global const &xyz) {
    auto f_col = std::forward<Callable>(func)(xyz);
    Mat1xN b_row = proj->GlobalToBasisValues(xyz);
    Coeff prod = f_col * b_row;
    return prod;
  }, proj->gauss());
}

template <typename Projection>
auto GetAverage(const Projection &proj) {
  auto const &mat_a = proj.basis().coeff();
  typename Projection::Value col_0 = proj.coeff().col(0);
  col_0 *= mat_a(0, 0);
  return col_0;
}

}  // namespace projection

template <std::floating_point Scalar, int kDimensions, int kDegrees,
    int kComponents>
class ProjectionWrapper;

/**
 * @brief A vector-valued function projected onto an given orthonormal basis.
 * 
 * @tparam S the data type of scalar components
 * @tparam kDimensions the dimension of the underlying physical space
 * @tparam kDegrees the degree of completeness
 * @tparam kComponents the number of function components
 */
template <std::floating_point S, int kDimensions, int kDegrees,
    int kComponents>
class Projection {
 public:
  static constexpr bool kLocal = false;
  using Scalar = S;
  using Wrapper = ProjectionWrapper<Scalar, kDimensions, kDegrees, kComponents>;
  using Basis = basis::OrthoNormal<Scalar, kDimensions, kDegrees>;
  using Taylor = typename Basis::Taylor;
  static constexpr int N = Basis::N;
  static constexpr int K = kComponents;
  static constexpr int P = kDegrees;
  using Gauss = typename Basis::Gauss;
  using Local = typename Gauss::Local;
  using Global = typename Gauss::Global;
  using MatNx1 = typename Basis::MatNx1;
  using Mat3xN = algebra::Matrix<Scalar, 3, N>;
  using Mat1xN = algebra::Matrix<Scalar, 1, N>;
  using Coeff = algebra::Matrix<Scalar, K, N>;
  using Value = algebra::Matrix<Scalar, K, 1>;
  using Gradient = algebra::Matrix<Scalar, 3, K>;

  using GaussOnLine = gauss::Legendre<Scalar, kDegrees + 1>;

 public:
  Coeff coeff_;
  Basis basis_;

 public:
  explicit Projection(const Gauss &gauss)
      : basis_(gauss) {
  }
  Projection() = default;
  Projection(const Projection &) = default;
  Projection &operator=(const Projection &) = default;
  Projection &operator=(Projection &&that) noexcept = default;
  Projection(Projection &&that) noexcept = default;
  ~Projection() noexcept = default;

  Mat1xN GlobalToBasisValues(Global const &global) const {
    return basis_(global);
  }
  Value GlobalToValue(Global const &global) const {
    return coeff_ * basis_(global);
  }
  Value operator()(Global const &global) const {
    return GlobalToValue(global);
  }
  /**
   * @brief Get the value of \f$ u(x,y,z) \f$ at a Gaussian point.
   * 
   * @param i the index of the Gaussian point
   * @return Value the value
   */
  Value GetValue(int i) const {
    auto &global = gauss().GetGlobalCoord(i);
    return GlobalToValue(global);
  }
  void LocalToGlobalAndValue(Local const &local,
      Global *global, Value *value) const {
    *global = gauss().lagrange().LocalToGlobal(local);
    *value = GlobalToValue(*global);
  }
  Coeff GetCoeffOnTaylorBasis() const {
    return coeff_ * basis_.coeff();
  }
  Basis const &basis() const {
    return basis_;
  }
  Gauss const &gauss() const {
    return basis().gauss();
  }
  Global const &center() const {
    return basis().center();
  }
  Coeff const &coeff() const {
    return coeff_;
  }
  Coeff &coeff() {
    return coeff_;
  }
  Value average() const {
    return projection::GetAverage(*this);
  }
  Mat3xN GlobalToBasisGradients(Global const &global) const {
    return basis_.GetGradValue(global).transpose();
  }
  /**
   * @brief Get \f$ \begin{bmatrix}\partial_{x}\\ \partial_{y}\\ \cdots \end{bmatrix} u \f$ at a Gaussian point.
   * 
   */
  Gradient GetGlobalGradient(int i) const {
    auto &global = gauss().GetGlobalCoord(i);
    Mat3xN basis_grad = GlobalToBasisGradients(global);
    return basis_grad * coeff().transpose();
  }

  template <typename Callable>
  void Approximate(Callable &&func) {
    projection::Project(this, std::forward<Callable>(func));
  }
  const Scalar *GetCoeffFrom(const Scalar *input) {
    std::memcpy(coeff_.data(), input, sizeof(Scalar) * coeff_.size());
    return input + coeff_.size();
  }
  Scalar *WriteCoeffTo(Scalar *output) const {
    std::memcpy(output, coeff_.data(), sizeof(Scalar) * coeff_.size());
    return output + coeff_.size();
  }
  static void AddCoeffTo(Coeff const &coeff, Scalar *output) {
    for (int c = 0; c < N; ++c) {
      for (int r = 0; r < K; ++r) {
        *output++ += coeff(r, c);
      }
    }
  }
  static void MinusCoeff(Coeff const &coeff, Scalar *output) {
    for (int c = 0; c < N; ++c) {
      for (int r = 0; r < K; ++r) {
        *output++ -= coeff(r, c);
      }
    }
  }
};

/**
 * @brief A light-weighted wrapper of `Projection`.
 * 
 * The object is light-weighted in the sense that it holds a pointer to a `basis::OrthoNormal` object and assumes the basis does not change in the lifetime of this object.
 * 
 * @tparam S the data type of scalar components
 * @tparam kDimensions the dimension of the underlying physical space
 * @tparam kDegrees the degree of completeness
 * @tparam kComponents the number of function components
 */
template <std::floating_point S, int kDimensions, int kDegrees,
    int kComponents>
class ProjectionWrapper {
 public:
  using Base = Projection<S, kDimensions, kDegrees, kComponents>;
  using Scalar = typename Base::Scalar;
  using Basis = typename Base::Basis;
  using Taylor = typename Base::Taylor;
  static constexpr int N = Base::N;
  static constexpr int K = Base::K;
  static constexpr int P = Base::P;
  using Gauss = typename Base::Gauss;
  using Local = typename Base::Local;
  using Global = typename Base::Global;
  using MatNx1 = typename Base::MatNx1;
  using Mat1xN = typename Base::Mat1xN;
  using MatKxK = algebra::Matrix<Scalar, K, K>;
  using Coeff = typename Base::Coeff;
  using Value = typename Base::Value;

 public:
  Coeff coeff_;
  const Basis *basis_ptr_ = nullptr;

 public:
  explicit ProjectionWrapper(const Basis &basis)
      : basis_ptr_(&basis) {
  }
  explicit ProjectionWrapper(const Base &that)
      : coeff_(that.coeff()), basis_ptr_(&(that.basis())) {
  }

  ProjectionWrapper() = default;
  ProjectionWrapper(const ProjectionWrapper &) = default;
  ProjectionWrapper &operator=(const ProjectionWrapper &) = default;
  ProjectionWrapper &operator=(ProjectionWrapper &&that) noexcept = default;
  ProjectionWrapper(ProjectionWrapper &&that) noexcept = default;
  ~ProjectionWrapper() noexcept = default;

  Basis const &basis() const {
    return *basis_ptr_;
  }
  Gauss const &gauss() const {
    return basis().gauss();
  }
  Coeff GetCoeffOnTaylorBasis() const {
    return coeff_ * basis().coeff();
  }
  Coeff const &coeff() const {
    return coeff_;
  }
  Coeff &coeff() {
    return coeff_;
  }
  Global const &center() const {
    return basis().center();
  }
  Value average() const {
    return projection::GetAverage(*this);
  }
  Mat1xN GlobalToBasisValues(Global const &global) const {
    return basis()(global);
  }
  template <typename Callable>
  void Approximate(Callable &&func) {
    projection::Project(this, std::forward<Callable>(func));
  }
  ProjectionWrapper &LeftMultiply(const MatKxK &left) {
    Coeff temp = left * coeff_;
    coeff_ = temp;
    return *this;
  }
  ProjectionWrapper &operator*=(Scalar ratio) {
    coeff_ *= ratio;
    return *this;
  }
  ProjectionWrapper &operator/=(Scalar ratio) {
    coeff_ /= ratio;
    return *this;
  }
  ProjectionWrapper &operator*=(const Value& ratio) {
    for (int i = 0; i < K; ++i) {
      coeff_.row(i) *= ratio[i];
    }
    return *this;
  }
  ProjectionWrapper &operator+=(Value offset) {
    offset /= basis().coeff()(0, 0);
    coeff_.col(0) += offset;
    return *this;
  }
  ProjectionWrapper &operator+=(const ProjectionWrapper &that) {
    assert(this->basis_ptr_ == that.basis_ptr_);
    coeff_ += that.coeff_;
    return *this;
  }
};

}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_PROJECTION_HPP_
