// Copyright 2019 PEI Weicheng and YANG Minghao

#ifndef MINI_RIEMANN_EULER_AUSM_HPP_
#define MINI_RIEMANN_EULER_AUSM_HPP_

#include <array>
#include <algorithm>
#include <cmath>

#include "mini/riemann/euler/types.hpp"

namespace mini {
namespace riemann {
namespace euler {

template <class G, int D>
class Ausm {
 public:
  constexpr static int kComponents = D + 2;
  constexpr static int kDimensions = D;
  // Types:
  using Gas = G;
  using Scalar = typename Gas::Scalar;
  using Flux = FluxTuple<Scalar, D>;
  using Conservative = Conservatives<Scalar, D>;
  using Primitive = Primitives<Scalar, D>;
  using Vector = typename Primitive::Vector;
  using Speed = Scalar;
  // Get F on T Axia
  Flux GetFluxUpwind(const Primitive& left, const Primitive& right) {
    Flux flux_positive = GetPositiveFlux(left);
    Flux flux_negative = GetNegativeFlux(right);
    flux_positive += flux_negative;
    return flux_positive;
  }
  // Get F of U
  static Flux GetFlux(const Primitive& state) {
    return Gas::PrimitiveToFlux(state);
  }

 private:
  static Flux BuildFlux(const Primitive& primitive, Scalar enthalpy)
      requires(kDimensions == 1) {
    return { 1, primitive.u(), enthalpy };
  }
  static Flux BuildFlux(const Primitive& primitive, Scalar enthalpy)
      requires(kDimensions == 2) {
    return { 1, primitive.u(), primitive.v(), enthalpy };
  }
  static Flux BuildFlux(const Primitive& primitive, Scalar enthalpy)
      requires(kDimensions == 3) {
    return { 1, primitive.u(), primitive.v(), primitive.w(), enthalpy };
  }
  Flux GetPositiveFlux(const Primitive& state) {
    double p_positive = state.p();
    double a = Gas::GetSpeedOfSound(state);
    double mach = state.u() / a;
    double mach_positive = mach;
    double h = a * a / Gas::GammaMinusOne() + state.GetKineticEnergy();
    Flux flux = BuildFlux(state, h);
    if (mach >= -1 && mach <= 1) {
      mach_positive = (mach + 1) * (mach + 1) * 0.25;
      p_positive = state.p() * (mach + 1) * 0.5;
    } else if (mach < -1) {
      mach_positive = 0.0;
      p_positive = 0.0;
    }
    double temp = state.rho() * a * mach_positive;
    flux *= temp;
    flux.momentumX() += p_positive;
    return flux;
  }
  Flux GetNegativeFlux(const Primitive& state) {
    double p_negative = state.p();
    double a = Gas::GetSpeedOfSound(state);
    double mach = state.u() / a;
    double mach_negative = mach;
    double h = a * a / Gas::GammaMinusOne() + state.GetKineticEnergy();
    Flux flux = BuildFlux(state, h);
    if (mach >= -1 && mach <= 1) {
      mach_negative = - (mach - 1) * (mach - 1) * 0.25;
      p_negative = - state.p() * (mach - 1) * 0.5;
    } else if (mach > 1) {
      mach_negative = 0.0;
      p_negative = 0.0;
    }
    double temp = state.rho() * a * mach_negative;
    flux *= temp;
    flux.momentumX() += p_negative;
    return flux;
  }
};

}  //  namespace euler
}  //  namespace riemann
}  //  namespace mini

#endif  //  MINI_RIEMANN_EULER_AUSM_HPP_
