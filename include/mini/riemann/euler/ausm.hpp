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

template <class GasType, int kDimensions>
class Ausm;

template <class GasType>
class Ausm<GasType, 1> {
 public:
  constexpr static int kComponents = 3;
  constexpr static int kDimensions = 1;
  // Types:
  using Gas = GasType;
  using Scalar = typename Gas::Scalar;
  using Flux = FluxTuple<Scalar, 1>;
  using Conservative = Conservatives<Scalar, 1>;
  using Primitive = Primitives<Scalar, 1>;
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
  Flux GetPositiveFlux(const Primitive& state) {
    double p_positive   = state.p();
    double a = Gas::GetSpeedOfSound(state);
    double mach = state.u() / a;
    double mach_positive = mach;
    double h = a * a / Gas::GammaMinusOne() + state.GetKineticEnergy();
    Flux flux = {1, state.u(), h};
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
    Flux flux = {1, state.u(), h};
    if (mach >= -1 && mach <= 1) {
      mach_negative = - (mach - 1) * (mach - 1) * 0.25;
      p_negative = - state.p() * (mach - 1) * 0.5;
    } else if (mach > 1) {
      mach_negative = 0.0;
      p_negative = 0.0;
    }
    flux *= state.rho() * a * mach_negative;
    flux.momentumX() += p_negative;
    return flux;
  }
};

template <class GasType>
class Ausm<GasType, 2> {
 public:
  constexpr static int kComponents = 4;
  constexpr static int kDimensions = 2;
  // Types:
  using Gas = GasType;
  using Scalar = typename Gas::Scalar;
  using Flux = FluxTuple<Scalar, 2>;
  using Conservative = Conservatives<Scalar, 2>;
  using Primitive = Primitives<Scalar, 2>;
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
  Flux GetPositiveFlux(const Primitive& state) {
    double p_positive = state.p();
    double a = Gas::GetSpeedOfSound(state);
    double mach = state.u() / a;
    double mach_positive = mach;
    double h = a * a / Gas::GammaMinusOne() + state.GetKineticEnergy();
    Flux flux = {1, state.u(), state.v(), h};
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
    Flux flux = {1, state.u(), state.v(), h};
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

template <class GasType>
class Ausm<GasType, 3> {
 public:
  constexpr static int kComponents = 5;
  constexpr static int kDimensions = 3;
  // Types:
  using Gas = GasType;
  using Scalar = typename Gas::Scalar;
  using Flux = FluxTuple<Scalar, 3>;
  using Conservative = Conservatives<Scalar, 3>;
  using Primitive = Primitives<Scalar, 3>;
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
  Flux GetPositiveFlux(const Primitive& state) {
    double p_positive = state.p();
    double a = Gas::GetSpeedOfSound(state);
    double mach = state.u() / a;
    double mach_positive = mach;
    double h = a * a / Gas::GammaMinusOne() + state.GetKineticEnergy();
    Flux flux = {1, state.u(), state.v(), state.w(), h};
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
    Flux flux = {1, state.u(), state.v(), state.w(), h};
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
