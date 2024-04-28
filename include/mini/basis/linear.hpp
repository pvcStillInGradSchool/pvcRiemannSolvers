//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_BASIS_LINEAR_HPP_
#define MINI_BASIS_LINEAR_HPP_

#include <concepts>

#include <cassert>
#include <cmath>

#include <iostream>
#include <type_traits>

#include "mini/algebra/eigen.hpp"

#include "mini/integrator/function.hpp"
#include "mini/integrator/face.hpp"
#include "mini/integrator/cell.hpp"

#include "mini/basis/taylor.hpp"

namespace mini {
namespace basis {

/**
 * @brief A basis of the linear space formed by polynomials less than or equal to a given degree.
 * 
 * @tparam Scalar the data type of scalar components
 * @tparam kDimensions the dimension of the underlying physical space
 * @tparam kDegrees the degree of completeness
 */
template <std::floating_point Scalar, int kDimensions, int kDegrees>
class Linear {
 public:
  using Taylor = basis::Taylor<Scalar, kDimensions, kDegrees>;
  static constexpr int N = Taylor::N;
  using Coord = typename Taylor::Coord;
  using MatNx1 = typename Taylor::MatNx1;
  using MatNxN = algebra::Matrix<Scalar, N, N>;
  using Integrator = std::conditional_t<kDimensions == 2,
      integrator::Face<Scalar, 2>, integrator::Cell<Scalar>>;

 public:
  explicit Linear(Coord const &center)
      : center_(center) {
    coeff_.setIdentity();
  }
  Linear() {
    center_.setZero();
    coeff_.setIdentity();
  }
  Linear(const Linear &) = default;
  Linear(Linear &&) noexcept = default;
  Linear &operator=(const Linear &) = default;
  Linear &operator=(Linear &&) noexcept = default;
  ~Linear() noexcept = default;

  MatNx1 operator()(Coord const &point) const {
    MatNx1 col = Taylor::GetValue(point - center_);
    MatNx1 res = algebra::GetLowerTriangularView(coeff_) * col;
    return res;
  }
  Coord const &center() const {
    return center_;
  }
  MatNxN const &coeff() const {
    return coeff_;
  }
  void Transform(MatNxN const &a) {
    MatNxN temp = a * coeff_;
    coeff_ = temp;
  }
  void Transform(algebra::LowerTriangularView<MatNxN> const &a) {
    MatNxN temp;
    algebra::GetLowerTriangularView(&temp) = a * coeff_;
    algebra::GetLowerTriangularView(&coeff_) = temp;
  }
  void Shift(const Coord &new_center) {
    center_ = new_center;
  }

 private:
  Coord center_;
  MatNxN coeff_;
};

template <std::floating_point Scalar, int kDimensions, int kDegrees>
class OrthoNormal {
 public:
  using Taylor = basis::Taylor<Scalar, kDimensions, kDegrees>;
  using Linear = basis::Linear<Scalar, kDimensions, kDegrees>;
  static constexpr int N = Linear::N;
  using Coord = typename Linear::Coord;
  using Integrator = typename Linear::Integrator;
  using MatNx1 = typename Linear::MatNx1;
  using MatNxN = typename Linear::MatNxN;
  using MatNxD = algebra::Matrix<Scalar, N, kDimensions>;

 public:
  explicit OrthoNormal(const Integrator &integrator)
      : integrator_ptr_(&integrator), basis_(integrator.center()) {
    assert(integrator.PhysDim() == kDimensions);
    OrthoNormalize(&basis_, integrator);
  }
  OrthoNormal() = default;
  OrthoNormal(const OrthoNormal &) = default;
  OrthoNormal(OrthoNormal &&) noexcept = default;
  OrthoNormal &operator=(const OrthoNormal &) = default;
  OrthoNormal &operator=(OrthoNormal &&) noexcept = default;
  ~OrthoNormal() noexcept = default;

  Coord const &center() const {
    return basis_.center();
  }
  MatNxN const &coeff() const {
    return basis_.coeff();
  }
  Integrator const &integrator() const {
    return *integrator_ptr_;
  }
  MatNx1 operator()(const Coord &global) const {
    auto local = global;
    local -= center();
    MatNx1 col = Taylor::GetValue(local);
    return coeff() * col;
  }
  Scalar Measure() const {
    auto v = basis_.coeff()(0, 0);
    return 1 / (v * v);
  }
  MatNxD GetGradValue(const Coord &global) const {
    auto local = global;
    local -= center();
    return Taylor::GetGradValue(local, coeff());
  }

 private:
  Integrator const *integrator_ptr_;
  Linear basis_;
};

}  // namespace basis
}  // namespace mini

#endif  // MINI_BASIS_LINEAR_HPP_
