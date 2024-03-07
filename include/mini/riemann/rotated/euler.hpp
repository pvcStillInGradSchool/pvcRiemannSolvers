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
  static Flux GetFlux(const Primitive& state) {
    return Base::GetFlux(state);
  }
  static FluxMatrix GetFluxMatrix(Conservative const& conservative) {
    return Gas::GetFluxMatrix(conservative);
  }
  Flux GetFluxOnSupersonicInlet(Conservative const& conservative) const {
    auto primitive = Gas::ConservativeToPrimitive(conservative);
    GlobalToNormal(&primitive);
    auto flux = unrotated_euler_.GetFlux(primitive);
    NormalToGlobal(&flux);
    return flux;
  }
  Flux GetFluxUpwind(Conservative const& left,
      Conservative const& right) const {
    auto left__primitive = Gas::ConservativeToPrimitive(left);
    auto right_primitive = Gas::ConservativeToPrimitive(right);
    GlobalToNormal(&left__primitive);
    GlobalToNormal(&right_primitive);
    auto flux = unrotated_euler_.GetFluxUpwind(
        left__primitive, right_primitive);
    NormalToGlobal(&flux);
    return flux;
  }
  Flux GetFluxOnInviscidWall(Conservative const& conservative) const {
    auto primitive = Gas::ConservativeToPrimitive(conservative);
    Flux flux; flux.setZero();
    flux.momentumX() = primitive.p();
    NormalToGlobal(&flux);
    return flux;
  }
  Flux GetFluxOnSupersonicOutlet(Conservative const& conservative) const {
    auto primitive = Gas::ConservativeToPrimitive(conservative);
    GlobalToNormal(&primitive);
    auto flux = unrotated_euler_.GetFlux(primitive);
    NormalToGlobal(&flux);
    return flux;
  }
  Flux GetFluxOnSubsonicInlet(Conservative const& conservative_i,
      Value const& given_value) const {
    Primitive primitive = Gas::ConservativeToPrimitive(conservative_i);
    auto p = primitive.energy();
    auto mach = Gas::GetMachFromPressure(p, given_value[0]);
    auto T = Gas::TotalTemperatureToTemperature(mach, given_value[4]);
    auto rho = p / Gas::R() / T;
    primitive.mass() = rho;
    primitive.energy() = p;
    auto uvw = mach * Gas::GetSpeedOfSound(T);
    auto v_cos = given_value[2];
    auto w_cos = given_value[3];
    // At any point of an inlet, the normal vector always points outward,
    // but the fluid always flows in, thus a negative sign is needed:
    auto u_cos = -std::sqrt(1 - v_cos * v_cos - w_cos * w_cos);
    primitive.momentumX() = uvw * u_cos;
    primitive.momentumY() = uvw * v_cos;
    primitive.momentumZ() = uvw * w_cos;
    // GlobalToNormal(&primitive);
    auto flux = unrotated_euler_.GetFlux(primitive);
    NormalToGlobal(&flux);
    std::cout << mach << "\n" << primitive.transpose() << "\n" << flux.transpose() << "\n";
    return flux;
  }
  Flux GetFluxOnSubsonicInletOld(Conservative const& conservative_i,
      Conservative const& conservative_o) const {
    auto primitive_i = Gas::ConservativeToPrimitive(conservative_i);
    auto primitive_o = Gas::ConservativeToPrimitive(conservative_o);
    Primitive primitive_b = primitive_o;
    Scalar u_nu_o = primitive_o.momentum().dot(normal());
    Scalar u_nu_i = primitive_i.momentum().dot(normal());
    Scalar u_nu_jump = u_nu_o - u_nu_i;
    Scalar a_i = Gas::GetSpeedOfSound(primitive_i);
    Scalar rho_a_i = primitive_i.rho() * (u_nu_o > 0 ? a_i : -a_i);
    primitive_b.p() = (primitive_i.p() + primitive_o.p()
        + rho_a_i * u_nu_jump) * 0.5;
    Scalar p_jump = primitive_o.p() - primitive_b.p();
    primitive_b.rho() -= p_jump / (a_i * a_i);
    primitive_b.momentum() += (p_jump / rho_a_i) * normal();
    GlobalToNormal(&primitive_b);
    auto flux = unrotated_euler_.GetFlux(primitive_b);
    NormalToGlobal(&flux);
    return flux;
  }
  Flux GetFluxOnSubsonicOutlet(Conservative const& conservative_i,
      Value const& given_value) const {
    Primitive primitive = Gas::ConservativeToPrimitive(conservative_i);
    primitive.energy() = given_value[4];
    GlobalToNormal(&primitive);
    auto flux = unrotated_euler_.GetFlux(primitive);
    NormalToGlobal(&flux);
    return flux;
  }
  Flux GetFluxOnSubsonicOutletOld(Conservative const& conservative_i,
      Conservative const& conservative_o) const {
    auto primitive_i = Gas::ConservativeToPrimitive(conservative_i);
    auto primitive_o = Gas::ConservativeToPrimitive(conservative_o);
    Primitive primitive_b = primitive_i;
    primitive_b.p() = primitive_o.p();
    Scalar p_jump = primitive_i.p() - primitive_b.p();
    Scalar a_i = Gas::GetSpeedOfSound(primitive_i);
    primitive_b.rho() -= p_jump / (a_i * a_i);
    Scalar u_nu_i = primitive_i.momentum().dot(normal());
    Scalar rho_a_i = primitive_i.rho() * (u_nu_i > 0 ? a_i : -a_i);
    primitive_b.momentum() += (p_jump / rho_a_i) * normal();
    GlobalToNormal(&primitive_b);
    auto flux = unrotated_euler_.GetFlux(primitive_b);
    NormalToGlobal(&flux);
    return flux;
  }
  Flux GetFluxOnSmartBoundary(Conservative const& conservative_i,
      Conservative const& conservative_o) const {
    auto primitive = Gas::ConservativeToPrimitive(conservative_i);
    Scalar a = Gas::GetSpeedOfSound(primitive);
    Scalar u_nu = primitive.momentum().dot(normal());
    Flux flux;
    if (u_nu < 0) {  // inlet
      if (u_nu + a < 0) {
        flux = GetFluxOnSupersonicInlet(conservative_o);
      } else {
        flux = GetFluxOnSubsonicInlet(conservative_i, conservative_o);
      }
    } else {  // outlet
      if (u_nu - a > 0) {
        flux = GetFluxOnSupersonicOutlet(conservative_i);
      } else {
        flux = GetFluxOnSubsonicOutlet(conservative_i, conservative_o);
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
