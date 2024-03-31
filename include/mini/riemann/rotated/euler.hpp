//  Copyright 2021 PEI Weicheng and YANG Minghao and JIANG Yuyan

#ifndef MINI_RIEMANN_ROTATED_EULER_HPP_
#define MINI_RIEMANN_ROTATED_EULER_HPP_

#include <concepts>
#include <type_traits>
#include <utility>
#include <iostream>

#include "mini/algebra/eigen.hpp"
#include "mini/riemann/euler/eigen.hpp"
#include "mini/constant/index.hpp"

namespace mini {
namespace riemann {
namespace rotated {

using namespace mini::constant::index;

template <class UnrotatedEuler>
class Euler {
  using Base = UnrotatedEuler;

 public:
  constexpr static int kComponents = Base::kComponents;
  constexpr static int kDimensions = Base::kDimensions;
  using Gas = typename Base::Gas;
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using Frame = std::array<Vector, kDimensions>;
  using Flux = typename Base::Flux;
  using FluxMatrix = typename Flux::FluxMatrix;
  using Conservative = typename Base::Conservative;
  using Primitive = typename Base::Primitive;
  using Value = typename Flux::Base;

  void Rotate(const Frame &frame) {
    frame_ = &frame;
  }
  Vector const &normal() const {
    return a();
  }

 private:
  const Vector &a() const {
    return (*frame_)[A];
  }
  const Vector &b() const {
    return (*frame_)[B];
  }
  const Vector &c() const requires(kDimensions == 3) {
    return (*frame_)[C];
  }
  Scalar a(int i) const {
    return a()[i];
  }
  Scalar b(int i) const {
    return b()[i];
  }
  Scalar c(int i) const requires(kDimensions == 3) {
    return c()[i];
  }
  Flux GlobalPrimitiveToGlobalFlux(Primitive *primitive) const {
    GlobalToNormal(primitive);
    auto flux = unrotated_euler_.GetFlux(*primitive);
    NormalToGlobal(&flux);
    return flux;
  }

 public:
  void GlobalToNormal(Value* v) const requires(kDimensions == 2) {
    Vector &p = v->momentum();
    Scalar p_a = p[X] * a(X) + p[Y] * a(Y);
    Scalar p_b = p[X] * b(X) + p[Y] * b(Y);
    p[A] = p_a;
    p[B] = p_b;
  }
  void NormalToGlobal(Value* v) const requires(kDimensions == 2) {
    Vector &p = v->momentum();
    Scalar p_x = p[A] * a(X) + p[B] * b(X);
    Scalar p_y = p[A] * a(Y) + p[B] * b(Y);
    p[X] = p_x;
    p[Y] = p_y;
  }
  void GlobalToNormal(Value* v) const requires(kDimensions == 3) {
    Vector &p = v->momentum();
    Scalar p_a = p.dot(a());
    Scalar p_b = p.dot(b());
    Scalar p_c = p.dot(c());
    p[A] = p_a;
    p[B] = p_b;
    p[C] = p_c;
  }
  void NormalToGlobal(Value* v) const requires(kDimensions == 3) {
    Vector &p = v->momentum();
    Scalar p_x = p[A] * a(X) + p[B] * b(X) + p[C] * c(X);
    Scalar p_y = p[A] * a(Y) + p[B] * b(Y) + p[C] * c(Y);
    Scalar p_z = p[A] * a(Z) + p[B] * b(Z) + p[C] * c(Z);
    p[X] = p_x;
    p[Y] = p_y;
    p[Z] = p_z;
  }
  static Flux GetFlux(const Primitive& primitive) {
    return Base::GetFlux(primitive);
  }
  static FluxMatrix GetFluxMatrix(Conservative const& conservative) {
    return Gas::GetFluxMatrix(conservative);
  }
  Flux GetFluxOnSupersonicInlet(Conservative const& conservative) const {
    auto primitive = Gas::ConservativeToPrimitive(conservative);
    return GlobalPrimitiveToGlobalFlux(&primitive);
  }
  Flux GetFluxOnSupersonicOutlet(Conservative const& conservative) const {
    auto primitive = Gas::ConservativeToPrimitive(conservative);
    return GlobalPrimitiveToGlobalFlux(&primitive);
  }

  Flux GetFluxUpwind(
      Conservative const& conservative__left,
      Conservative const& conservative_right) const {
    auto primitive__left = Gas::ConservativeToPrimitive(conservative__left);
    auto primitive_right = Gas::ConservativeToPrimitive(conservative_right);
    GlobalToNormal(&primitive__left);
    GlobalToNormal(&primitive_right);
    auto flux = unrotated_euler_.GetFluxUpwind(
        primitive__left, primitive_right);
    NormalToGlobal(&flux);
    return flux;
  }
  Flux GetFluxOnInviscidWall(Conservative const& conservative) const {
    auto primitive__left = Gas::ConservativeToPrimitive(conservative);
    GlobalToNormal(&primitive__left);
    auto primitive_right = primitive__left;
    primitive_right.u() = -primitive__left.u();
    auto flux = unrotated_euler_.GetFluxUpwind(
        primitive__left, primitive_right);
    NormalToGlobal(&flux);
    return flux;
  }

  /**
   * @brief Get the Flux on subsonic inlet.
   * 
   * See "2.7 Subsonic Inflow BC" in Carlson (2011).
   */
  Flux GetFluxOnSubsonicInlet(Conservative const& conservative_interior,
      Value const& given_value) const {
    Primitive primitive_interior = Gas::ConservativeToPrimitive(conservative_interior);
    auto c_interior = Gas::GetSpeedOfSound(primitive_interior);
    GlobalToNormal(&primitive_interior);
    auto riemann_interior = -primitive_interior.u()
        - c_interior * Gas::GammaMinusOneUnderTwo();
    auto u_cos = given_value[1];
    auto v_cos = given_value[2];
    auto w_cos = given_value[3];
    auto T_total = given_value[4];
    // TODO(PVC): cache A since it's constant.
    auto A = a(X) * u_cos + a(Y) * v_cos + a(Z) * w_cos;
    assert(A < 0.0);
    auto square_of = [](Scalar x) { return x * x; };
    auto square_of_A = square_of(A);
    auto gamma_times_R = Gas::Gamma() * Gas::R();
    auto larger = [](Scalar a, Scalar b_half, Scalar c) -> Scalar {
      auto sqrt_delta = std::sqrt(b_half * b_half - a * c);
      auto x_1 = (-b_half + sqrt_delta) / a;
      auto x_2 = (-b_half - sqrt_delta) / a;
      return std::max(x_1, x_2);
    };
    auto c_boundary = larger(
        /* the a in (a x^2 + b x + c = 0) */
        square_of_A + Gas::GammaMinusOneUnderTwo(),
        /* 1/2 of the b in (a x^2 + b x + c = 0) */
        riemann_interior,
        /* the c in (a x^2 + b x + c = 0) */
        Gas::GammaMinusOneOverTwo() * square_of(riemann_interior)
            - square_of_A * gamma_times_R * T_total);
    auto T_boundary = square_of(c_boundary) / gamma_times_R;
    auto mach_boundary = Gas::GetMachFromTemperature(T_boundary, T_total);
    auto p_total = given_value[0];
    auto p_boundary = Gas::TotalPressureToPressure(mach_boundary, p_total);
    auto &primitive_boundary = primitive_interior;
    primitive_boundary.p() = p_boundary;
    primitive_boundary.rho() = p_boundary / Gas::R() / T_boundary;
    auto speed_boundary = mach_boundary * c_boundary;
    primitive_boundary.u() = u_cos * speed_boundary;
    primitive_boundary.v() = v_cos * speed_boundary;
    primitive_boundary.w() = w_cos * speed_boundary;
    return GlobalPrimitiveToGlobalFlux(&primitive_boundary);
  }
  Flux GetFluxOnSubsonicInletOld(
      Conservative const& conservative_interior,
      Conservative const& conservative_exterior) const {
    auto primitive_interior = Gas::ConservativeToPrimitive(conservative_interior);
    auto primitive_exterior = Gas::ConservativeToPrimitive(conservative_exterior);
    Primitive primitive_boundary = primitive_exterior;
    Scalar u_normal_exterior = primitive_exterior.momentum().dot(normal());
    Scalar u_normal_interior = primitive_interior.momentum().dot(normal());
    Scalar u_normal_jump = u_normal_exterior - u_normal_interior;
    Scalar c_interior = Gas::GetSpeedOfSound(primitive_interior);
    Scalar rho_c_interior = primitive_interior.rho() * (u_normal_exterior > 0 ? c_interior : -c_interior);
    primitive_boundary.p() = (primitive_interior.p() + primitive_exterior.p()
        + rho_c_interior * u_normal_jump) * 0.5;
    Scalar p_jump = primitive_exterior.p() - primitive_boundary.p();
    primitive_boundary.rho() -= p_jump / (c_interior * c_interior);
    primitive_boundary.momentum() += (p_jump / rho_c_interior) * normal();
    return GlobalPrimitiveToGlobalFlux(&primitive_boundary);
  }
  /**
   * @brief Get the Flux on subsonic outlet.
   * 
   * See "2.4 Pressure Outflow BC" in Carlson (2011).
   */
  Flux GetFluxOnSubsonicOutlet(Conservative const& conservative_interior,
      Value const& given_value) const {
    Primitive primitive = Gas::ConservativeToPrimitive(conservative_interior);
    auto RT = primitive.p() / primitive.rho();
    primitive.p() = given_value[4];
    primitive.rho() = primitive.p() / RT;
    return GlobalPrimitiveToGlobalFlux(&primitive);
  }
  Flux GetFluxOnSubsonicOutletOld(Conservative const& conservative_interior,
      Conservative const& conservative_exterior) const {
    auto primitive_interior = Gas::ConservativeToPrimitive(conservative_interior);
    auto primitive_exterior = Gas::ConservativeToPrimitive(conservative_exterior);
    Primitive primitive_boundary = primitive_interior;
    primitive_boundary.p() = primitive_exterior.p();
    Scalar p_jump = primitive_interior.p() - primitive_boundary.p();
    Scalar c_interior = Gas::GetSpeedOfSound(primitive_interior);
    primitive_boundary.rho() -= p_jump / (c_interior * c_interior);
    Scalar u_normal_interior = primitive_interior.momentum().dot(normal());
    Scalar rho_c_interior = primitive_interior.rho() * (u_normal_interior > 0 ? c_interior : -c_interior);
    primitive_boundary.momentum() += (p_jump / rho_c_interior) * normal();
    return GlobalPrimitiveToGlobalFlux(&primitive_boundary);
  }
  Flux GetFluxOnSmartBoundary(
      Conservative const& conservative_interior,
      Conservative const& conservative_exterior) const {
    return GetFluxUpwind(conservative_interior, conservative_exterior);
    auto primitive = Gas::ConservativeToPrimitive(conservative_interior);
    Scalar a = Gas::GetSpeedOfSound(primitive);
    Scalar u_normal = primitive.momentum().dot(normal());
    Flux flux;
    if (u_normal < 0) {  // inlet
      if (u_normal + a < 0) {
        flux = GetFluxOnSupersonicInlet(conservative_exterior);
      } else {
        flux = GetFluxOnSubsonicInlet(conservative_interior, conservative_exterior);
      }
    } else {  // outlet
      if (u_normal - a > 0) {
        flux = GetFluxOnSupersonicOutlet(conservative_interior);
      } else {
        flux = GetFluxOnSubsonicOutlet(conservative_interior, conservative_exterior);
      }
    }
    return flux;
  }

 private:
  using EigenMatrices = riemann::euler::EigenMatrices<Gas>;
  EigenMatrices eigen_matrices_;
  UnrotatedEuler unrotated_euler_;
  Frame const *frame_;

 public:
  using Matrix = typename EigenMatrices::Mat5x5;
  void UpdateEigenMatrices(const Conservative& big_u) {
    eigen_matrices_ = EigenMatrices(big_u, a(), b(), c());
  }
  const Matrix& L() const {
    return eigen_matrices_.L;
  }
  const Matrix& R() const {
    return eigen_matrices_.R;
  }
};

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  //  MINI_RIEMANN_ROTATED_EULER_HPP_
