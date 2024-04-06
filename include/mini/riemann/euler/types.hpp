//  Copyright 2019 PEI Weicheng and YANG Minghao
#ifndef MINI_RIEMANN_EULER_TYPES_HPP_
#define MINI_RIEMANN_EULER_TYPES_HPP_

#include <concepts>

#include <cassert>
#include <cmath>
#include <cstring>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace riemann {
namespace euler {

template <std::floating_point ScalarType, int kDimensions>
class Tuple : public algebra::Vector<ScalarType, kDimensions+2> {
  using Base = algebra::Vector<ScalarType, kDimensions+2>;

 public:
  // Types:
  using Scalar = ScalarType;
  using Vector = algebra::Vector<Scalar, kDimensions>;
  // Constructors:
  using Base::Base;
  // Accessors and Mutators:
  Scalar mass() const {
    return (*this)[0];
  }
  const Vector& momentum() const {
    return *reinterpret_cast<const Vector*>(&((*this)[1]));
  }
  Scalar momentumX() const {
    return (*this)[1];
  }
  Scalar momentumY() const {
    assert(kDimensions >= 2);
    return (*this)[2];
  }
  Scalar momentumZ() const {
    assert(kDimensions == 3);
    return (*this)[3];
  }
  Scalar energy() const {
    return (*this)[kDimensions+1];
  }
  Scalar& mass() {
    return (*this)[0];
  }
  Vector& momentum() {
    return *reinterpret_cast<Vector*>(&momentumX());
  }
  Scalar& momentumX() {
    return (*this)[1];
  }
  Scalar& momentumY() {
    assert(kDimensions >= 2);
    return (*this)[2];
  }
  Scalar& momentumZ() {
    assert(kDimensions == 3);
    return (*this)[3];
  }
  Scalar& energy() {
    return (*this)[kDimensions+1];
  }

  static void VelocityToMomentum(Scalar rho, Tuple *tuple)
      requires(kDimensions == 1) {
    tuple->momentumX() *= rho;
  }
  static void MomentumToVelocity(Scalar rho, Tuple *tuple)
      requires(kDimensions == 1) {
    tuple->momentumX() /= rho;
  }
  static void VelocityToMomentum(Scalar rho, Tuple *tuple)
      requires(kDimensions == 2) {
    tuple->momentumX() *= rho;
    tuple->momentumY() *= rho;
  }
  static void MomentumToVelocity(Scalar rho, Tuple *tuple)
      requires(kDimensions == 2) {
    tuple->momentumX() /= rho;
    tuple->momentumY() /= rho;
  }
  static void VelocityToMomentum(Scalar rho, Tuple *tuple)
      requires(kDimensions == 3) {
    tuple->momentumX() *= rho;
    tuple->momentumY() *= rho;
    tuple->momentumZ() *= rho;
  }
  static void MomentumToVelocity(Scalar rho, Tuple *tuple)
      requires(kDimensions == 3) {
    tuple->momentumX() /= rho;
    tuple->momentumY() /= rho;
    tuple->momentumZ() /= rho;
  }
};

template <std::floating_point ScalarType, int kDimensions>
class FluxTuple : public Tuple<ScalarType, kDimensions> {
 public:
  // Types:
  using Base = Tuple<ScalarType, kDimensions>;
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using FluxMatrix = algebra::Matrix<Scalar, kDimensions + 2, kDimensions>;

  // Constructors:
  using Base::Base;
};

template <std::floating_point ScalarType, int kDimensions>
class Primitives : public Tuple<ScalarType, kDimensions> {
  using Base = Tuple<ScalarType, kDimensions>;

 public:
  // Types:
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  // Constructors:
  using Base::Base;
  // Accessors and Mutators:
  Scalar rho() const {
    return this->mass();
  }
  Scalar u() const {
    return this->momentumX();
  }
  Scalar v() const {
    return this->momentumY();
  }
  Scalar w() const {
    return this->momentumZ();
  }
  const Vector& velocity() const {
    return this->momentum();
  }
  Scalar p() const {
    return this->energy();
  }
  Scalar& rho() {
    return this->mass();
  }
  Scalar& u() {
    return this->momentumX();
  }
  Scalar& v() {
    return this->momentumY();
  }
  Scalar& w() {
    return this->momentumZ();
  }
  Scalar& p() {
    return this->energy();
  }
  Vector& velocity() {
    return this->momentum();
  }
  Scalar GetKineticEnergy() const requires(kDimensions == 1) {
    return 0.5 * (u() * u());
  }
  Scalar GetKineticEnergy() const requires(kDimensions == 2) {
    return 0.5 * (u() * u() + v() * v());
  }
  Scalar GetKineticEnergy() const requires(kDimensions == 3) {
    return 0.5 * (u() * u() + v() * v() + w() * w());
  }
  Scalar GetDynamicPressure() const {
    return rho() * GetKineticEnergy();
  }
};

template <std::floating_point ScalarType, int kDimensions>
struct Conservatives : public Tuple<ScalarType, kDimensions> {
  using Base = Tuple<ScalarType, kDimensions>;

  // Types:
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  // Constructors:
  using Base::Base;
};

template <std::floating_point ScalarType, double kGamma>
class IdealGas {
 public:
  using Scalar = ScalarType;

 private:
  static_assert(kGamma >= 1);

  template <typename Primitive>
  static void SetZeroIfNegative(Primitive *primitive) {
    if (primitive->rho() < 0) {
      primitive->setZero();
    }
    if (primitive->p() < 0) {
      primitive->setZero();
    }
  }

 public:
  // Constants:
  static constexpr Scalar R() {
    return 287.05;
  }
  static constexpr Scalar Gamma() {
    return kGamma;
  }
  static constexpr Scalar OneOverGamma() {
    return 1 / Gamma();
  }
  static constexpr Scalar GammaPlusOne() {
    return Gamma() + 1;
  }
  static constexpr Scalar GammaPlusOneOverTwo() {
    return GammaPlusOne() / 2;
  }
  static constexpr Scalar GammaPlusOneOverFour() {
    return GammaPlusOne() / 4;
  }
  static constexpr Scalar GammaMinusOne() {
    return Gamma() - 1;
  }
  static constexpr Scalar OneOverGammaMinusOne() {
    return 1 / GammaMinusOne();
  }
  static constexpr Scalar GammaOverGammaMinusOne() {
    return Gamma() / GammaMinusOne();
  }
  static constexpr Scalar GammaMinusOneOverGamma() {
    return GammaMinusOne() / Gamma();
  }
  static constexpr Scalar GammaMinusOneOverTwo() {
    return GammaMinusOne() / 2;
  }
  static constexpr Scalar GammaMinusOneUnderTwo() {
    return 2 / GammaMinusOne();
  }
  static constexpr Scalar Cv() {
    return R() * OneOverGammaMinusOne();
  }
  static constexpr Scalar Cp() {
    return R() * GammaOverGammaMinusOne();
  }
  // Converters:
  static constexpr Scalar GetSpeedOfSound(Scalar temperature) {
    return (temperature > 0) ? std::sqrt(Gamma() * R() * temperature) : 0;
  }
  static constexpr Scalar GetSpeedOfSound(Scalar rho, Scalar p) {
    return (rho > 0 && p > 0) ? std::sqrt(Gamma() * p / rho) : 0;
  }
  template <int kDimensions>
  static Scalar GetSpeedOfSound(Primitives<Scalar, kDimensions> const& state) {
    return GetSpeedOfSound(state.rho(), state.p());
  }

  static constexpr Scalar GetMachFactor(Scalar mach) {
    return 1 + GammaMinusOneOverTwo() * mach * mach;
  }
  static constexpr Scalar TotalTemperatureToTemperature(Scalar mach,
      Scalar total_temperature) {
    auto factor = GetMachFactor(mach);
    return total_temperature / factor;
  }
  static constexpr Scalar TotalPressureToPressure(Scalar mach,
      Scalar total_pressure) {
    auto factor = std::pow(GetMachFactor(mach), GammaOverGammaMinusOne());
    return total_pressure / factor;
  }
  static constexpr Scalar GetMachFromTemperatureRatio(Scalar ratio) {
    return std::sqrt((ratio - 1) / GammaMinusOneOverTwo());
  }
  static constexpr Scalar GetMachFromTemperature(Scalar temperature,
      Scalar total_temperature) {
    Scalar ratio = total_temperature / temperature;
    return GetMachFromTemperatureRatio(ratio);
  }
  static constexpr Scalar GetMachFromPressure(Scalar pressure,
      Scalar total_pressure) {
    Scalar ratio = total_pressure / pressure;
    ratio = std::pow(ratio, GammaMinusOneOverGamma());
    return GetMachFromTemperatureRatio(ratio);
  }

  template <int kDimensions>
  static Primitives<Scalar, kDimensions> ConservativeToPrimitive(
      Conservatives<Scalar, kDimensions> const &conservative) {
    auto primitive = Primitives<Scalar, kDimensions>(conservative);
    SetZeroIfNegative(&primitive);
    if (primitive.rho()) {
      Tuple<Scalar, kDimensions>::MomentumToVelocity(primitive.rho(),
          &primitive);
      primitive.energy() -= primitive.GetDynamicPressure();
      primitive.energy() *= GammaMinusOne();
      SetZeroIfNegative(&primitive);
    }
    return primitive;
  }
  template <int kDimensions>
  static Conservatives<Scalar, kDimensions> PrimitiveToConservative(
      Primitives<Scalar, kDimensions> const &primitive) {
    auto conservative = Conservatives<Scalar, kDimensions>(primitive);
    Tuple<Scalar, kDimensions>::VelocityToMomentum(primitive.rho(),
        &conservative);
    conservative.energy() *= OneOverGammaMinusOne();  // p / (gamma - 1)
    conservative.energy() += primitive.GetDynamicPressure();
    return conservative;
  }
  template <int kDimensions>
  static FluxTuple<Scalar, kDimensions> PrimitiveToFlux(
      const Primitives<Scalar, kDimensions> &primitive) {
    auto conservative = PrimitiveToConservative(primitive);
    conservative *= primitive.u();
    auto flux_x = FluxTuple<Scalar, kDimensions>(conservative);
    flux_x.momentumX() += primitive.p();
    flux_x.energy() += primitive.p() * primitive.u();
    return flux_x;
  }
  static auto GetFluxMatrix(Conservatives<Scalar, 3> const& cv) {
    using FluxMatrix = typename FluxTuple<Scalar, 3>::FluxMatrix;
    FluxMatrix mat;
    auto pv = ConservativeToPrimitive(cv);
    auto rho = pv.rho(), u = pv.u(), v = pv.v(), w = pv.w(), p = pv.p();
    auto rho_u = cv.momentumX(), rho_v = cv.momentumY(), rho_w = cv.momentumZ();
    auto rho_h0 = cv.energy() + p;
    mat.col(0) << rho_u, rho_u * u + p, rho_v * u, rho_w * u, rho_h0 * u;
    mat.col(1) << rho_v, rho_u * v, rho_v * v + p, rho_w * v, rho_h0 * v;
    mat.col(2) << rho_w, rho_u * w, rho_v * w, rho_w * w + p, rho_h0 * w;
    return mat;
  }
};

}  // namespace euler
}  // namespace riemann
}  // namespace mini

#endif  //  MINI_RIEMANN_EULER_TYPES_HPP_
