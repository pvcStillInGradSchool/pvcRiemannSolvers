//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_GAUSS_TRIANGLE_HPP_
#define MINI_GAUSS_TRIANGLE_HPP_

#include <concepts>

#include <cassert>
#include <cmath>

#include <algorithm>
#include <cstring>
#include <type_traits>

#include "mini/gauss/face.hpp"
#include "mini/geometry/triangle.hpp"

namespace mini {
namespace gauss {

/**
 * @brief Numerical integrators on triangular elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam kPhysDim  Dimension of the physical space.
 * @tparam kPoints  Number of qudrature points.
 */
template <std::floating_point Scalar, int kPhysDim, int kPoints>
class Triangle : public Face<Scalar, kPhysDim> {
  static constexpr int D = kPhysDim;

 public:
  using Lagrange = geometry::Triangle<Scalar, kPhysDim>;
  using Real = typename Lagrange::Real;
  using Local = typename Lagrange::Local;
  using Global = typename Lagrange::Global;
  using Jacobian = typename Lagrange::Jacobian;
  using Frame = typename Lagrange::Frame;

 private:
  static const std::array<Local, kPoints> local_coords_;
  static const std::array<Scalar, kPoints> local_weights_;
  std::array<Global, kPoints> global_coords_;
  std::array<Scalar, kPoints> global_weights_;
  std::array<Frame, kPoints> normal_frames_;
  Lagrange const *lagrange_;
  Scalar area_;

 public:
  int CountPoints() const final {
    return kPoints;
  }

 public:
  const Global &GetGlobalCoord(int i) const final {
    assert(0 <= i && i < CountPoints());
    return global_coords_[i];
  }
  const Scalar &GetGlobalWeight(int i) const final {
    assert(0 <= i && i < CountPoints());
    return global_weights_[i];
  }
  const Local &GetLocalCoord(int i) const final {
    assert(0 <= i && i < CountPoints());
    return local_coords_[i];
  }
  const Scalar &GetLocalWeight(int i) const final {
    assert(0 <= i && i < CountPoints());
    return local_weights_[i];
  }

 protected:
  Global &GetGlobalCoord(int i) final {
    assert(0 <= i && i < CountPoints());
    return global_coords_[i];
  }
  Scalar &GetGlobalWeight(int i) final {
    assert(0 <= i && i < CountPoints());
    return global_weights_[i];
  }

 public:
  const Frame &GetNormalFrame(int i) const final {
    assert(0 <= i && i < CountPoints());
    return normal_frames_[i];
  }
  Frame &GetNormalFrame(int i) final {
    assert(0 <= i && i < CountPoints());
    return normal_frames_[i];
  }

 public:
  explicit Triangle(Lagrange const &lagrange)
      : lagrange_(&lagrange) {
    area_ = this->BuildQuadraturePoints();
    _NormalFrameBuilder<Scalar, kPhysDim>::Build(this);
  }

  const Lagrange &lagrange() const final {
    return *lagrange_;
  }

  Scalar area() const final {
    return area_;
  }
};

template <std::floating_point Scalar, int kPhysDim, int kPoints>
class _TriangleBuilder;

template <std::floating_point Scalar, int kPhysDim, int kPoints>
std::array<typename Triangle<Scalar, kPhysDim, kPoints>::Local,
    kPoints> const
Triangle<Scalar, kPhysDim, kPoints>::local_coords_
    = _TriangleBuilder<Scalar, kPhysDim, kPoints>::BuildLocalCoords();

template <std::floating_point Scalar, int kPhysDim, int kPoints>
const std::array<Scalar, kPoints>
Triangle<Scalar, kPhysDim, kPoints>::local_weights_
    = _TriangleBuilder<Scalar, kPhysDim, kPoints>::BuildLocalWeights();

template <std::floating_point Scalar, int kPhysDim, int kPoints>
class _TriangleBuilder {
  using Local =
      typename Triangle<Scalar, kPhysDim, kPoints>::Local;

 public:
  static constexpr auto BuildLocalCoords() requires(kPoints == 1) {
    Scalar a = .3333333333333333333333333333333333;
    std::array<Local, kPoints> points;
    points[0] = { a, a };
    return points;
  }
  static constexpr auto BuildLocalWeights() requires(kPoints == 1) {
    std::array<Scalar, kPoints> weights{ 1.0 / 2.0 };
    return weights;
  }

  static constexpr auto BuildLocalCoords() requires(kPoints == 3) {
    std::array<Local, kPoints> points;
    int q = 0;
    // the only S21 orbits
    Scalar a_s21[] = { 1./6. };
    for (auto a : a_s21) {
      auto b = 1 - a - a;
      points[q++] = { a, a };
      points[q++] = { a, b };
      points[q++] = { b, a };
    }
    assert(q == kPoints);
    return points;
  }
  static constexpr auto BuildLocalWeights() requires(kPoints == 3) {
    std::array<Scalar, kPoints> weights;
    for (int q = 0; q < 3; ++q)
      weights[q] = 1./3.;
    for (int q = 0; q < kPoints; ++q)
      weights[q] /= 2.0;
    return weights;
  }

  static constexpr auto BuildLocalCoords() requires(kPoints == 6) {
    std::array<Local, kPoints> points;
    int q = 0;
    // the two S21 orbits
    Scalar a_s21[] = {
        .44594849091596488631832925388305199,
        .09157621350977074345957146340220151 };
    for (auto a : a_s21) {
      auto b = 1 - a - a;
      points[q++] = { a, a };
      points[q++] = { a, b };
      points[q++] = { b, a };
    }
    assert(q == kPoints);
    return points;
  }
  static constexpr auto BuildLocalWeights() requires(kPoints == 6) {
    std::array<Scalar, kPoints> weights;
    for (int q = 0; q < 3; ++q)
      weights[q] = .22338158967801146569500700843312280;
    for (int q = 3; q < 6; ++q)
      weights[q] = .10995174365532186763832632490021053;
    for (int q = 0; q < kPoints; ++q)
      weights[q] /= 2.0;
    return weights;
  }

  static constexpr auto BuildLocalCoords() requires(kPoints == 12) {
    std::array<Local, kPoints> points;
    int q = 0;
    // the two S21 orbits
    Scalar a_s21[] = {
        .06308901449150222834033160287081916,
        .24928674517091042129163855310701908 };
    for (auto a : a_s21) {
      auto b = 1 - a - a;
      points[q++] = { a, a };
      points[q++] = { a, b };
      points[q++] = { b, a };
    }
    {  // the only S111 orbit
      Scalar a = .05314504984481694735324967163139815;
      Scalar b = .31035245103378440541660773395655215;
      Scalar c = 1 - a - b;
      points[q++] = { a, b };
      points[q++] = { a, c };
      points[q++] = { b, a };
      points[q++] = { b, c };
      points[q++] = { c, a };
      points[q++] = { c, b };
    }
    assert(q == kPoints);
    return points;
  }
  static constexpr auto BuildLocalWeights() requires(kPoints == 12) {
    std::array<Scalar, kPoints> weights;
    for (int q = 0; q < 3; ++q)
      weights[q] = .05084490637020681692093680910686898;
    for (int q = 3; q < 6; ++q)
      weights[q] = .11678627572637936602528961138557944;
    for (int q = 6; q < 12; ++q)
      weights[q] = .08285107561837357519355345642044245;
    for (int q = 0; q < kPoints; ++q)
      weights[q] /= 2.0;
    return weights;
  }

  static constexpr auto BuildLocalCoords() requires(kPoints == 16) {
    std::array<Local, kPoints> points;
    int q = 0;
    {  // the only S3 orbit
      Scalar a = .33333333333333333333333333333333333;
      points[q++] = { a, a };
    }
    // the three S21 orbits
    Scalar a_s21[] = {
        .17056930775176020662229350149146450,
        .05054722831703097545842355059659895,
        .45929258829272315602881551449416932 };
    for (auto a : a_s21) {
      auto b = 1 - a - a;
      points[q++] = { a, a };
      points[q++] = { a, b };
      points[q++] = { b, a };
    }
    {  // the six S111 orbits
      Scalar a = .26311282963463811342178578628464359;
      Scalar b = .00839477740995760533721383453929445;
      Scalar c = 1 - a - b;
      points[q++] = { a, b };
      points[q++] = { a, c };
      points[q++] = { b, a };
      points[q++] = { b, c };
      points[q++] = { c, a };
      points[q++] = { c, b };
    }
    assert(q == kPoints);
    return points;
  }
  static constexpr auto BuildLocalWeights() requires(kPoints == 16) {
    std::array<Scalar, kPoints> weights;
    for (int q = 0; q < 1; ++q)
      weights[q] = .14431560767778716825109111048906462;
    for (int q = 1; q < 4; ++q)
      weights[q] = .10321737053471825028179155029212903;
    for (int q = 4; q < 7; ++q)
      weights[q] = .03245849762319808031092592834178060;
    for (int q = 7; q < 10; ++q)
      weights[q] = .09509163426728462479389610438858432;
    for (int q = 10; q < 16; ++q)
      weights[q] = .02723031417443499426484469007390892;
    for (int q = 0; q < 16; ++q)
      weights[q] /= 2.0;
    return weights;
  }
};

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_TRIANGLE_HPP_
