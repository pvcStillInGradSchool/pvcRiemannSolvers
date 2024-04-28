//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_INTEGRATOR_TETRAHEDRON_HPP_
#define MINI_INTEGRATOR_TETRAHEDRON_HPP_

#include <concepts>

#include <cassert>
#include <cmath>
#include <cstring>

#include <algorithm>
#include <type_traits>
#include <utility>

#include "mini/integrator/cell.hpp"
#include "mini/coordinate/tetrahedron.hpp"

namespace mini {
namespace integrator {

/**
 * @brief Numerical integrators on tetrahedral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam kPoints  Number of qudrature points, which must be chosen from { 1, 4, 14, 15, 24, 46 }.
 */
template <std::floating_point Scalar, int kPoints>
class Tetrahedron : public Cell<Scalar> {
  static_assert(kPoints == 1 || kPoints == 4 || kPoints == 14 || kPoints == 15
      || kPoints == 24 || kPoints == 46);

 public:
  using Coordinate = coordinate::Tetrahedron<Scalar>;
  using Real = typename Coordinate::Real;
  using Local = typename Coordinate::Local;
  using Global = typename Coordinate::Global;
  using Jacobian = typename Coordinate::Jacobian;

 private:
  static const std::array<Local, kPoints> local_coords_;
  static const std::array<Scalar, kPoints> local_weights_;
  std::array<Global, kPoints> global_coords_;
  std::array<Scalar, kPoints> global_weights_;
  Coordinate const *coordinate_;
  Scalar volume_;

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
  explicit Tetrahedron(Coordinate const &lagrange)
      : coordinate_(&lagrange) {
    volume_ = this->BuildQuadraturePoints();
  }
  Tetrahedron(const Tetrahedron &) = default;
  Tetrahedron &operator=(const Tetrahedron &) = default;
  Tetrahedron(Tetrahedron &&) noexcept = default;
  Tetrahedron &operator=(Tetrahedron &&) noexcept = default;
  virtual ~Tetrahedron() noexcept = default;

  const Coordinate &coordinate() const final {
    return *coordinate_;
  }

  Scalar volume() const final {
    return volume_;
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
    Scalar a = 0.25;
    Points points;
    points[0] = { a, a, a };
    return points;
  }
  static constexpr Weights _BuildLocalWeights() requires(kPoints == 1) {
    Weights weights{ 1.0 / 6.0 };
    return weights;
  }

  static constexpr Points _BuildLocalCoords() requires(kPoints == 4) {
    Points points;
    int q = 0;
    // the only S31 orbit
    Scalar a = 0.13819660112501051517954131656343619;
    auto c = 1 - 3 * a;
    points[q++] = { a, a, a };
    points[q++] = { a, a, c };
    points[q++] = { a, c, a };
    points[q++] = { c, a, a };
    assert(q == kPoints);
    return points;
  }
  static constexpr Weights _BuildLocalWeights() requires(kPoints == 4) {
    Weights weights;
    for (int q = 0; q < kPoints; ++q)
      weights[q] = 0.25 / 6.0;
    return weights;
  }

  static constexpr Points _BuildLocalCoords() requires(kPoints == 14) {
    Points points;
    int q = 0;
    // the two S31 orbits
    Scalar a_s31[] = {
        0.31088591926330060979734573376345783,
        0.09273525031089122640232391373703061 };
    for (auto a : a_s31) {
      auto c = 1 - 3 * a;
      points[q++] = { a, a, a };
      points[q++] = { a, a, c };
      points[q++] = { a, c, a };
      points[q++] = { c, a, a };
    }
    {  // the only S22 orbit
      Scalar a = 0.04550370412564964949188052627933943;
      auto c = (1 - 2 * a) * .5;
      points[q++] = { a, a, c };
      points[q++] = { a, c, a };
      points[q++] = { a, c, c };
      points[q++] = { c, a, a };
      points[q++] = { c, a, c };
      points[q++] = { c, c, a };
    }
    assert(q == kPoints);
    return points;
  }
  static constexpr Weights _BuildLocalWeights() requires(kPoints == 14) {
    Weights weights;
    for (int q = 0; q < 4; ++q)
      weights[q] = 0.11268792571801585079918565233328633;
    for (int q = 4; q < 8; ++q)
      weights[q] = 0.07349304311636194954371020548632750;
    for (int q = 8; q < kPoints; ++q)
      weights[q] = 0.04254602077708146643806942812025744;
    for (int q = 0; q < kPoints; ++q)
      weights[q] /= 6.0;
    return weights;
  }

  static constexpr Points _BuildLocalCoords() requires(kPoints == 15) {
    Points points;
    int q = 0;
    {  // the only S4 orbit
      Scalar a = 0.25;
      points[q++] = { a, a, a };
    }
    // the two S31 orbits
    Scalar a_s31[] = {
        0.09197107805272303,
        0.31979362782962991 };
    for (auto a : a_s31) {
      auto c = 1 - 3 * a;
      points[q++] = { a, a, a };
      points[q++] = { a, a, c };
      points[q++] = { a, c, a };
      points[q++] = { c, a, a };
    }
    {  // the only S22 orbit
      Scalar a = 0.05635083268962916;
      auto c = (1 - 2 * a) * .5;
      points[q++] = { a, a, c };
      points[q++] = { a, c, a };
      points[q++] = { a, c, c };
      points[q++] = { c, a, a };
      points[q++] = { c, a, c };
      points[q++] = { c, c, a };
    }
    assert(q == kPoints);
    return points;
  }
  static constexpr Weights _BuildLocalWeights() requires(kPoints == 15) {
    Weights weights;
    for (int q = 0; q < 1; ++q)
      weights[q] = 16.0 / 135.0;
    for (int q = 1; q < 5; ++q)
      weights[q] = 0.07193708377901862;
    for (int q = 5; q < 9; ++q)
      weights[q] = 0.06906820722627239;
    for (int q = 9; q < kPoints; ++q)
      weights[q] = 20.0 / 378.0;
    for (int q = 0; q < kPoints; ++q)
      weights[q] /= 6.0;
    return weights;
  }

  static constexpr Points _BuildLocalCoords() requires(kPoints == 24) {
    Points points;
    int q = 0;
    // the three S31 orbits
    Scalar a_s31[] = {
        0.21460287125915202928883921938628499,
        0.04067395853461135311557944895641006,
        0.32233789014227551034399447076249213 };
    for (auto a : a_s31) {
      auto c = 1 - 3 * a;
      points[q++] = { a, a, a };
      points[q++] = { a, a, c };
      points[q++] = { a, c, a };
      points[q++] = { c, a, a };
    }
    {  // the only S211 orbit
      Scalar a = 0.06366100187501752529923552760572698;
      Scalar b = 0.60300566479164914136743113906093969;
      auto c = 1 - a - a - b;
      points[q++] = { a, a, b };
      points[q++] = { a, a, c };
      points[q++] = { a, b, a };
      points[q++] = { a, b, c };
      points[q++] = { a, c, a };
      points[q++] = { a, c, b };
      points[q++] = { b, a, a };
      points[q++] = { b, a, c };
      points[q++] = { b, c, a };
      points[q++] = { c, a, a };
      points[q++] = { c, a, b };
      points[q++] = { c, b, a };
    }
    assert(q == kPoints);
    return points;
  }
  static constexpr Weights _BuildLocalWeights() requires(kPoints == 24) {
    Weights weights;
    for (int q = 0; q < 4; ++q)
      weights[q] = 0.03992275025816749209969062755747998;
    for (int q = 4; q < 8; ++q)
      weights[q] = 0.01007721105532064294801323744593686;
    for (int q = 8; q < 12; ++q)
      weights[q] = 0.05535718154365472209515327785372602;
    for (int q = 12; q < 24; ++q)
      weights[q] = 27./560.;
    for (int q = 0; q < 24; ++q)
      weights[q] /= 6.0;
    return weights;
  }

  static constexpr Points _BuildLocalCoords() requires(kPoints == 46) {
    Points points;
    int q = 0;
    // the four S31 orbits
    Scalar a_s31[] = {
        .03967542307038990126507132953938949,
        .31448780069809631378416056269714830,
        .10198669306270330000000000000000000,
        .18420369694919151227594641734890918 };
    for (auto a : a_s31) {
      auto c = 1 - 3 * a;
      points[q++] = { a, a, a };
      points[q++] = { a, a, c };
      points[q++] = { a, c, a };
      points[q++] = { c, a, a };
    }
    {  // the only S22 orbit
      Scalar a = .06343628775453989240514123870189827;
      auto c = (1 - 2 * a) * .5;
      points[q++] = { a, a, c };
      points[q++] = { a, c, a };
      points[q++] = { a, c, c };
      points[q++] = { c, a, a };
      points[q++] = { c, a, c };
      points[q++] = { c, c, a };
    }
    // the two S211 orbits
    std::pair<Scalar, Scalar> ab_s211[2]{
      { .02169016206772800480266248262493018,
        .71993192203946593588943495335273478 },
      { .20448008063679571424133557487274534,
        .58057719012880922417539817139062041 }};
    for (auto [a, b] : ab_s211) {
      auto c = 1 - a - a - b;
      points[q++] = { a, a, b };
      points[q++] = { a, a, c };
      points[q++] = { a, b, a };
      points[q++] = { a, b, c };
      points[q++] = { a, c, a };
      points[q++] = { a, c, b };
      points[q++] = { b, a, a };
      points[q++] = { b, a, c };
      points[q++] = { b, c, a };
      points[q++] = { c, a, a };
      points[q++] = { c, a, b };
      points[q++] = { c, b, a };
    }
    assert(q == kPoints);
    return points;
  }
  static constexpr Weights _BuildLocalWeights() requires(kPoints == 46) {
    Weights weights;
    for (int q = 0; q < 4; ++q)
      weights[q] = .00639714777990232132145142033517302;
    for (int q = 4; q < 8; ++q)
      weights[q] = .04019044802096617248816115847981783;
    for (int q = 8; q < 12; ++q)
      weights[q] = .02430797550477032117486910877192260;
    for (int q = 12; q < 16; ++q)
      weights[q] = .05485889241369744046692412399039144;
    for (int q = 16; q < 22; ++q)
      weights[q] = .03571961223409918246495096899661762;
    for (int q = 22; q < 34; ++q)
      weights[q] = .00718319069785253940945110521980376;
    for (int q = 34; q < 46; ++q)
      weights[q] = .01637218194531911754093813975611913;
    for (int q = 0; q < 46; ++q)
      weights[q] /= 6.0;
    return weights;
  }
};
template <std::floating_point Scalar, int kPoints>
typename Tetrahedron<Scalar, kPoints>::Points const
Tetrahedron<Scalar, kPoints>::local_coords_
    = Tetrahedron<Scalar, kPoints>::BuildLocalCoords();

template <std::floating_point Scalar, int kPoints>
typename Tetrahedron<Scalar, kPoints>::Weights const
Tetrahedron<Scalar, kPoints>::local_weights_
    = Tetrahedron<Scalar, kPoints>::BuildLocalWeights();

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_TETRAHEDRON_HPP_
