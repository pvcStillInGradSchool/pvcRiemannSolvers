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
class AdvectionUpstreamSplittingMethod {
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
    constexpr bool kPositive = false, kNegative = true;
    return GetSignedFlux<kPositive>(left) + GetSignedFlux<kNegative>(right);
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
  template <bool kNegative>
  Flux GetSignedFlux(const Primitive& state) {
    double a = Gas::GetSpeedOfSound(state);
    double h = a * a / Gas::GammaMinusOne() + state.GetKineticEnergy();
    Flux flux = BuildFlux(state, h);
    double mach = state.u() / a;
    double p = state.p();
    auto add_sign = [](Scalar x) { return kNegative ? -x : x; };
    auto signed_mach = (1 + add_sign(mach)) * 0.5;
    if (mach >= -1 && mach <= 1) {
      mach = add_sign(signed_mach * signed_mach);
      p *= signed_mach;
    } else if (signed_mach < 0) {
      mach = 0.0;
      p = 0.0;
    }
    flux *= state.rho() * a * mach;
    flux.momentumX() += p;
    return flux;
  }
};

}  //  namespace euler
}  //  namespace riemann
}  //  namespace mini

#endif  //  MINI_RIEMANN_EULER_AUSM_HPP_
