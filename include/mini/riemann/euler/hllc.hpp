// Copyright 2019 PEI Weicheng and YANG Minghao

#ifndef MINI_RIEMANN_EULER_HLLC_HPP_
#define MINI_RIEMANN_EULER_HLLC_HPP_

#include <array>
#include <algorithm>
#include <cmath>

#include "mini/riemann/euler/types.hpp"

namespace mini {
namespace riemann {
namespace euler {

template <class GasType, int D>
class Hllc {
 public:
  constexpr static int kDimensions = D;
  constexpr static int kComponents = kDimensions + 2;
  // Types:
  using Gas = GasType;
  using Scalar = typename Gas::Scalar;
  using Flux = FluxTuple<Scalar, kDimensions>;
  using Conservative = Conservatives<Scalar, kDimensions>;
  using Primitive = Primitives<Scalar, kDimensions>;
  using Vector = typename Primitive::Vector;
  using Speed = Scalar;
  // Get F on T Axia
  Flux GetFluxUpwind(const Primitive& left, const Primitive& right) {
    Initialize(left, right);
    Flux flux;
    if (0.0 <= wave_left_) {
      flux = GetFlux(left);
    } else if (wave_right_ <= 0.0) {
      flux = GetFlux(right);
    } else if (0.0 <= wave_star_) {
      flux = GetStarFlux(left, wave_left_);
    } else if (wave_star_ < 0.0) {
      flux = GetStarFlux(right, wave_right_);
    }
    return flux;
  }
  // Get F of U
  static Flux GetFlux(const Primitive& state) {
    return Gas::PrimitiveToFlux(state);
  }

 private:
  Speed wave_star_;
  Speed wave_left_;
  Speed wave_right_;
  void Initialize(const Primitive& left, const Primitive& right) {
    double rho_average = (left.rho() + right.rho()) / 2;
    double a_left = Gas::GetSpeedOfSound(left);
    double a_right = Gas::GetSpeedOfSound(right);
    double a_average = (a_left + a_right) / 2;
    double p_pvrs = (left.p() + right.p() - (right.u() - left.u()) *
                     rho_average * a_average) / 2;
    double p_estimate = std::max(0.0, p_pvrs);
    wave_left_ = left.u() - a_left * GetQ(p_estimate, left.p());
    wave_right_ = right.u() + a_right * GetQ(p_estimate, right.p());
    wave_star_ = (right.p() - left.p() +
                   left.rho() *  left.u() *  (wave_left_ -  left.u()) -
                  right.rho() * right.u() * (wave_right_ - right.u())) /
                  (left.rho() * (wave_left_  -  left.u()) -
                  right.rho() * (wave_right_ - right.u()));
  }
  double GetQ(Scalar const& p_estimate, Scalar const& p_k) {
    if (p_estimate <= p_k) {
      return 1.0;
    } else {
      double temp = 1 + Gas::GammaPlusOneOverTwo() * (p_estimate / p_k - 1) /
                    Gas::Gamma();
      return std::sqrt(temp);
    }
  }
  void SetMomentum(Scalar rho, const Primitive &primitive,
      Conservative *conservative) const requires(kDimensions == 1) {
    conservative->momentumX() = rho * wave_star_;
  }
  void SetMomentum(Scalar rho, const Primitive &primitive,
      Conservative *conservative) const requires(kDimensions == 2) {
    conservative->momentumX() = rho * wave_star_;
    conservative->momentumY() = rho * primitive.v();
  }
  void SetMomentum(Scalar rho, const Primitive &primitive,
      Conservative *conservative) const requires(kDimensions == 3) {
    conservative->momentumX() = rho * wave_star_;
    conservative->momentumY() = rho * primitive.v();
    conservative->momentumZ() = rho * primitive.w();
  }
  Flux GetStarFlux(const Primitive& state, Speed const& wave_k) {
    Flux flux = GetFlux(state);
    Conservative u_k = Gas::PrimitiveToConservative(state);
    double energy = u_k.energy();
    double temp = state.rho() * (wave_k - state.u()) / (wave_k - wave_star_);
    Conservative u_star_k;
    u_star_k.mass() = temp;
    SetMomentum(temp, state, &u_star_k);
    u_star_k.energy() = energy / state.rho() + (wave_star_ - state.u()) *
        (wave_star_ + state.p() / (state.rho() * (wave_k - state.u())));
    u_star_k.energy() *= temp;
    u_star_k -= u_k;
    u_star_k *= wave_k;
    flux += u_star_k;
    return flux;
  }
};

}  //  namespace euler
}  //  namespace riemann
}  //  namespace mini

#endif  //  MINI_RIEMANN_EULER_HLLC_HPP_
