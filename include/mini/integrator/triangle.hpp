//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_INTEGRATOR_TRIANGLE_HPP_
#define MINI_INTEGRATOR_TRIANGLE_HPP_

#include <concepts>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <memory>

#include "mini/integrator/face.hpp"
#include "mini/coordinate/triangle.hpp"

namespace mini {
namespace integrator {

/**
 * @brief Numerical integrators on triangular elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam kPhysDim  Dimension of the physical space.
 * @tparam kPoints  Number of qudrature points, which must be chosen from { 1, 3, 6, 12, 16 }.
 */
template <std::floating_point Scalar, int kPhysDim, int kPoints>
class Triangle : public Face<Scalar, kPhysDim> {
  static_assert(kPoints == 1 || kPoints == 3 || kPoints == 6 || kPoints == 12
      || kPoints == 16);
  static constexpr int D = kPhysDim;

 public:
  using Coordinate = coordinate::Triangle<Scalar, kPhysDim>;
  using Real = typename Coordinate::Real;
  using Local = typename Coordinate::Local;
  using Global = typename Coordinate::Global;
  using Jacobian = typename Coordinate::Jacobian;
  using Frame = typename Coordinate::Frame;

 private:
  using Base = Face<Scalar, kPhysDim>;
  static const std::array<Local, kPoints> local_coords_;
  static const std::array<Scalar, kPoints> local_weights_;
  std::array<Global, kPoints> global_coords_;
  std::array<Scalar, kPoints> global_weights_;
  std::array<Frame, kPoints> normal_frames_;
  Coordinate const *coordinate_;
  Scalar area_;

 public:
  int CountPoints() const final {
    return kPoints;
  }

 public:
  const Global &GetGlobal(int i) const final {
    assert(0 <= i && i < CountPoints());
    return global_coords_[i];
  }
  const Scalar &GetGlobalWeight(int i) const final {
    assert(0 <= i && i < CountPoints());
    return global_weights_[i];
  }
  const Local &GetLocal(int i) const final {
    assert(0 <= i && i < CountPoints());
    return local_coords_[i];
  }
  const Scalar &GetLocalWeight(int i) const final {
    assert(0 <= i && i < CountPoints());
    return local_weights_[i];
  }

 protected:
  Global &GetGlobal(int i) final {
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

 protected:
  Frame &GetNormalFrame(int i) final {
    assert(0 <= i && i < CountPoints());
    return normal_frames_[i];
  }

 public:
  std::unique_ptr<Base>
  Clone(typename Coordinate::Base const &coordinate) const final {
    return std::make_unique<Triangle>(
        dynamic_cast<Coordinate const &>(coordinate));
  }

  explicit Triangle(Coordinate const &lagrange)
      : coordinate_(&lagrange) {
    area_ = this->BuildQuadraturePoints();
    Base::BuildNormalFrames(this);
  }

  const Coordinate &coordinate() const final {
    return *coordinate_;
  }

  Scalar area() const final {
    return area_;
  }

 public:
  using Points = std::array<Local, kPoints>;
  using Weights = std::array<Scalar, kPoints>;

  static constexpr Points BuildLocalCoords() {
    return _BuildLocalCoords();
  }
  static constexpr Weights BuildLocalWeights() {
    return _BuildLocalWeights();
  }

 private:
  static constexpr Points _BuildLocalCoords() requires(kPoints == 1) {
    Scalar a = .3333333333333333333333333333333333;
    Points points;
    points[0] = { a, a };
    return points;
  }
  static constexpr Weights _BuildLocalWeights() requires(kPoints == 1) {
    Weights weights{ 1.0 / 2.0 };
    return weights;
  }

  static constexpr Points _BuildLocalCoords() requires(kPoints == 3) {
    Points points;
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
  static constexpr Weights _BuildLocalWeights() requires(kPoints == 3) {
    Weights weights;
    for (int q = 0; q < 3; ++q)
      weights[q] = 1./3.;
    for (int q = 0; q < kPoints; ++q)
      weights[q] /= 2.0;
    return weights;
  }

  static constexpr Points _BuildLocalCoords() requires(kPoints == 6) {
    Points points;
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
  static constexpr Weights _BuildLocalWeights() requires(kPoints == 6) {
    Weights weights;
    for (int q = 0; q < 3; ++q)
      weights[q] = .22338158967801146569500700843312280;
    for (int q = 3; q < 6; ++q)
      weights[q] = .10995174365532186763832632490021053;
    for (int q = 0; q < kPoints; ++q)
      weights[q] /= 2.0;
    return weights;
  }

  static constexpr Points _BuildLocalCoords() requires(kPoints == 12) {
    Points points;
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
  static constexpr Weights _BuildLocalWeights() requires(kPoints == 12) {
    Weights weights;
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

  static constexpr Points _BuildLocalCoords() requires(kPoints == 16) {
    Points points;
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
  static constexpr Weights _BuildLocalWeights() requires(kPoints == 16) {
    Weights weights;
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

template <std::floating_point Scalar, int kPhysDim, int kPoints>
typename Triangle<Scalar, kPhysDim, kPoints>::Points const
Triangle<Scalar, kPhysDim, kPoints>::local_coords_
    = Triangle<Scalar, kPhysDim, kPoints>::BuildLocalCoords();

template <std::floating_point Scalar, int kPhysDim, int kPoints>
typename Triangle<Scalar, kPhysDim, kPoints>::Weights const
Triangle<Scalar, kPhysDim, kPoints>::local_weights_
    = Triangle<Scalar, kPhysDim, kPoints>::BuildLocalWeights();

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_TRIANGLE_HPP_
