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
    auto positive = [](Scalar *mach_normal, Scalar *p_normal){
      Scalar mach = *mach_normal;
      if (mach >= -1 && mach <= 1) {
        auto half_mach_plus = (1 + mach) * 0.5;
        *mach_normal = half_mach_plus * half_mach_plus;
        *p_normal *= half_mach_plus;
      } else if (mach < -1) {
        *mach_normal = 0.0;
        *p_normal = 0.0;
      }
    };
    auto negative = [](Scalar *mach_normal, Scalar *p_normal){
      Scalar mach = *mach_normal;
      if (mach >= -1 && mach <= 1) {
        auto half_mach_minus = (1 - mach) * 0.5;
        *mach_normal = -half_mach_minus * half_mach_minus;
        *p_normal *= half_mach_minus;
      } else if (mach > 1) {
        *mach_normal = 0.0;
        *p_normal = 0.0;
      }
    };
    return GetSignedFlux(left, positive) + GetSignedFlux(right, negative);
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
  template <class SignedOperation>
  Flux GetSignedFlux(const Primitive& state, SignedOperation &&operation) {
    double a = Gas::GetSpeedOfSound(state);
    double h = a * a / Gas::GammaMinusOne() + state.GetKineticEnergy();
    Flux flux = BuildFlux(state, h);
    double mach_normal = state.u() / a;
    double p_normal = state.p();
    operation(&mach_normal, &p_normal);
    flux *= state.rho() * a * mach_normal;
    flux.momentumX() += p_normal;
    return flux;
  }
};

}  //  namespace euler
}  //  namespace riemann
}  //  namespace mini

#endif  //  MINI_RIEMANN_EULER_AUSM_HPP_
