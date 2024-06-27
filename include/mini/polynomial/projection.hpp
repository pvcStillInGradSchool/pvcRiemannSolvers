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
#include "mini/integrator/function.hpp"
#include "mini/basis/linear.hpp"
#include "mini/polynomial/expansion.hpp"

namespace mini {
namespace polynomial {

/**
 * @brief Implement methods common in Projection-like class.
 * 
 */
namespace {

template <typename Projection, typename Callable>
void Project(Projection *proj, Callable &&func) {
  using Global = typename Projection::Global;
  using Mat1xN = typename Projection::Mat1xN;
  using Value = typename Projection::Value;
  using Coeff = typename Projection::Coeff;
  using Return = std::invoke_result_t<Callable, Global>;
  static_assert(std::is_same_v<Return, Value> || std::is_scalar_v<Return>);
  Coeff coeff = integrator::Integrate([&](Global const &xyz) {
    auto f_col = func(xyz);
    Mat1xN b_row = proj->GlobalToBasisValues(xyz);
    Coeff prod = f_col * b_row;
    return prod;
  }, proj->integrator());
  proj->SetCoeff(coeff);
}

template <typename Projection>
auto GetAverage(const Projection &proj) {
  auto const &mat_a = proj.basis().coeff();
  typename Projection::Value col_0 = proj.coeff().col(0);
  col_0 *= mat_a(0, 0);
  return col_0;
}

}  // anonymous namespace

template <std::floating_point Scalar, int kDimensions, int kDegrees,
    int kComponents>
class ProjectionWrapper;

/**
 * @brief A vector-valued function projected onto an given `basis::OrthoNormal` basis.
 * 
 * @tparam S the data type of scalar components
 * @tparam kDimensions the dimension of the underlying physical space
 * @tparam kDegrees the degree of completeness
 * @tparam kComponents the number of function components
 */
template <std::floating_point S, int kDimensions, int kDegrees,
    int kComponents>
class Projection : public Expansion<kComponents,
    basis::OrthoNormal<S, kDimensions, kDegrees>> {
 public:
  static constexpr bool kLocal = false;
  using Scalar = S;
  using Wrapper = ProjectionWrapper<Scalar, kDimensions, kDegrees, kComponents>;
  using Basis = basis::OrthoNormal<Scalar, kDimensions, kDegrees>;
  using Taylor = typename Basis::Taylor;
  static constexpr int N = Basis::N;
  static constexpr int K = kComponents;
  static constexpr int P = kDegrees;
  static constexpr int D = kDimensions;
  using Integrator = typename Basis::Integrator;
  using Local = typename Integrator::Local;
  using Global = typename Integrator::Global;
  using MatNx1 = typename Basis::MatNx1;
  using Mat3xN = algebra::Matrix<Scalar, 3, N>;
  using Mat1xN = algebra::Matrix<Scalar, 1, N>;
  using Coeff = algebra::Matrix<Scalar, K, N>;
  using Value = algebra::Matrix<Scalar, K, 1>;
  using Gradient = algebra::Matrix<Scalar, 3, K>;

 private:
  Basis basis_;

  static constexpr int kFields = K * N;

 public:
  explicit Projection(const Integrator &integrator)
      : basis_(integrator) {
  }
  Projection() = default;
  Projection(const Projection &) = default;
  Projection &operator=(const Projection &) = default;
  Projection &operator=(Projection &&that) noexcept = default;
  Projection(Projection &&that) noexcept = default;
  ~Projection() noexcept = default;

  Projection const &projection() const {
    return *this;
  }

  Mat1xN GlobalToBasisValues(Global const &global) const {
    return basis_(global);
  }
  Value GlobalToValue(Global const &global) const {
    return this->coeff_ * basis_(global);
  }

  /**
   * @brief Get the value of \f$ u(x,y,z) \f$ at a Integratorian point.
   * 
   * @param i the index of the Integratorian point
   * @return Value the value
   */
  Value GetValue(int i) const {
    auto &global = integrator().GetGlobal(i);
    return GlobalToValue(global);
  }
  void LocalToGlobalAndValue(Local const &local,
      Global *global, Value *value) const {
    *global = integrator().coordinate().LocalToGlobal(local);
    *value = GlobalToValue(*global);
  }
  Coeff GetCoeffOnTaylorBasis() const {
    return this->coeff_ * basis_.coeff();
  }
  Basis const &basis() const {
    return basis_;
  }
  Integrator const &integrator() const {
    return basis().integrator();
  }
  Global const &center() const {
    return basis().center();
  }

  /**
   * @brief Set the projection coefficient of a given mode.
   * 
   * @param i the index of the mode
   * @param value the value \f$ \langle u | \phi_i \rangle \f$
   */
  void SetValue(int i, Value const &value) {
    this->coeff_.col(i) = value;
  }

  Value average() const {
    return GetAverage(*this);
  }
  Mat3xN GlobalToBasisGradients(Global const &global) const {
    return basis_.GetGradValue(global).transpose();
  }
  /**
   * @brief Get \f$ \begin{bmatrix}\partial_{x}\\ \partial_{y}\\ \cdots \end{bmatrix} u \f$ at a Integratorian point.
   * 
   */
  Gradient GetGlobalGradient(int i) const {
    auto &global = integrator().GetGlobal(i);
    Mat3xN basis_grad = GlobalToBasisGradients(global);
    return basis_grad * this->coeff_.transpose();
  }

  /**
   * @brief A wrapper of Projection::GetValue and Projection::GetGlobalGradient for reusing intermediate results.
   * 
   */
  std::pair<Value, Gradient> GetGlobalValueGradient(int i) const {
    auto &global = integrator().GetGlobal(i);
    Mat3xN basis_grad = GlobalToBasisGradients(global);
    return { GlobalToValue(global), basis_grad * this->coeff_.transpose() };
  }

  template <typename Callable>
  void Approximate(Callable &&func) {
    Project(this, std::forward<Callable>(func));
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
class ProjectionWrapper : public Expansion<kComponents,
    typename Projection<S, kDimensions, kDegrees, kComponents>::Basis> {
 public:
  using Base = Projection<S, kDimensions, kDegrees, kComponents>;
  using Scalar = typename Base::Scalar;
  using Basis = typename Base::Basis;
  using Taylor = typename Base::Taylor;
  static constexpr int N = Base::N;
  static constexpr int K = Base::K;
  static constexpr int P = Base::P;
  static constexpr int D = Base::D;
  using Integrator = typename Base::Integrator;
  using Local = typename Base::Local;
  using Global = typename Base::Global;
  using MatNx1 = typename Base::MatNx1;
  using Mat1xN = typename Base::Mat1xN;
  using MatKxK = algebra::Matrix<Scalar, K, K>;
  using Coeff = typename Base::Coeff;
  using Value = typename Base::Value;

 public:
  const Basis *basis_ptr_ = nullptr;

 public:
  explicit ProjectionWrapper(const Basis &basis)
      : basis_ptr_(&basis) {
  }

  explicit ProjectionWrapper(const Base &that)
      : basis_ptr_(&(that.basis())) {
    this->coeff_ = that.coeff();
  }

  ProjectionWrapper() = default;
  ProjectionWrapper(const ProjectionWrapper &) = default;
  ProjectionWrapper &operator=(const ProjectionWrapper &) = default;
  ProjectionWrapper &operator=(ProjectionWrapper &&that) noexcept = default;
  ProjectionWrapper(ProjectionWrapper &&that) noexcept = default;
  ~ProjectionWrapper() noexcept = default;

  ProjectionWrapper const &projection() const {
    return *this;
  }

  Basis const &basis() const {
    return *basis_ptr_;
  }
  Integrator const &integrator() const {
    return basis().integrator();
  }
  Coeff GetCoeffOnTaylorBasis() const {
    return this->coeff() * basis().coeff();
  }
  Global const &center() const {
    return basis().center();
  }
  Value average() const {
    return GetAverage(*this);
  }
  Mat1xN GlobalToBasisValues(Global const &global) const {
    return basis()(global);
  }
  Value GlobalToValue(Global const &global) const {
    return this->coeff() * GlobalToBasisValues(global);
  }

  template <typename Callable>
  void Approximate(Callable &&func) {
    Project(this, std::forward<Callable>(func));
  }
  ProjectionWrapper &operator+=(Value offset) {
    offset /= basis().coeff()(0, 0);
    this->coeff_.col(0) += offset;
    return *this;
  }
  ProjectionWrapper &operator+=(const ProjectionWrapper &that) {
    assert(this->basis_ptr_ == that.basis_ptr_);
    this->coeff_ += that.coeff_;
    return *this;
  }
};

}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_PROJECTION_HPP_
