// Copyright 2023 PEI Weicheng
#ifndef MINI_RIEMANN_DIFFUSIVE_NAVIER_STOKES_HPP_
#define MINI_RIEMANN_DIFFUSIVE_NAVIER_STOKES_HPP_

#include <utility>

#include "mini/algebra/eigen.hpp"
#include "mini/riemann/euler/types.hpp"
#include "mini/constant/index.hpp"

namespace mini {
namespace riemann {
namespace diffusive {

using namespace mini::constant::index;

template <typename G>
class NavierStokes {
 public:
  static constexpr int kDimensions = 3;
  static constexpr int kComponents = 5;
  using Gas = G;
  using Scalar = typename Gas::Scalar;
  using Gradient = algebra::Matrix<Scalar, kDimensions, kComponents>;
  using Flux = euler::FluxTuple<Scalar, kDimensions>;
  using FluxMatrix = typename Flux::FluxMatrix;
  using Value = algebra::Vector<Scalar, kComponents>;
  using FluxVector = Value;
  using Conservative = euler::Conservatives<Scalar, kDimensions>;
  using Primitive = euler::Primitives<Scalar, kDimensions>;
  using Vector = typename Primitive::Vector;
  using Tensor = algebra::Vector<Scalar, 6>;

  static void SetProperty(Scalar nu, Scalar prandtl) {
    nu_ = nu;
    prandtl_ = prandtl;
  }

  static std::pair<Scalar, Scalar> GetViscosity(Scalar rho) {
    Scalar mu = nu_ * rho;
    Scalar zeta = -2.0 / 3 * mu;  // Stokes' hypothesis
    return {mu, zeta};
  }

  static Scalar GetThermalConductivity(Scalar rho) {
    Scalar mu = nu_ * rho;
    return mu * Gas::Cp() / prandtl_;
  }

 protected:
  static Scalar nu_;
  static Scalar prandtl_;
  static constexpr int kMass = 0;
  static constexpr int U = 1 + X;
  static constexpr int V = 1 + Y;
  static constexpr int W = 1 + Z;
  static constexpr int kEnergy = 4;
  static constexpr int kPressure = 4;
  static constexpr int kTemperature = 4;

 public:
  static std::pair<Primitive, Gradient> ConservativeToPrimitive(
      Conservative const &c_val, Gradient const &c_grad) {
    auto p_val = Gas::ConservativeToPrimitive(c_val);
    Gradient p_grad;
    auto &&grad_u = p_grad.col(U);
    auto &&grad_v = p_grad.col(V);
    auto &&grad_w = p_grad.col(W);
    auto &grad_rho = c_grad.col(kMass);
    auto &grad_rho_u = c_grad.col(U);
    auto &grad_rho_v = c_grad.col(V);
    auto &grad_rho_w = c_grad.col(W);
    p_grad.col(kMass) = grad_rho;
    auto rho = p_val.mass();
    auto u = p_val.momentumX();
    auto v = p_val.momentumY();
    auto w = p_val.momentumZ();
    auto p = p_val.energy();
    grad_u = (grad_rho_u - u * grad_rho) / rho;
    grad_v = (grad_rho_v - v * grad_rho) / rho;
    grad_w = (grad_rho_w - w * grad_rho) / rho;
    auto rho_u = c_val.momentumX();
    auto rho_v = c_val.momentumY();
    auto rho_w = c_val.momentumZ();
    auto &&grad_p = p_grad.col(kPressure);
    grad_p  = u * grad_rho_u + rho_u * grad_u;
    grad_p += v * grad_rho_v + rho_v * grad_v;
    grad_p += w * grad_rho_w + rho_w * grad_w;
    grad_p *= -0.5;
    grad_p += c_grad.col(kEnergy);
    grad_p *= Gas::GammaMinusOne();
    auto &&grad_T = p_grad.col(kTemperature);
    grad_T = grad_p / rho - (p / (rho * rho)) * grad_rho;
    grad_T /= Gas::R();
    return {p_val, p_grad};
  }

  static Tensor GetViscousStressTensor(Gradient const &p_grad, Scalar rho) {
    Tensor tau;
    const auto &grad_u = p_grad.col(U);
    const auto &grad_v = p_grad.col(V);
    const auto &grad_w = p_grad.col(W);
    auto div_uvw = grad_u[X] + grad_v[Y] + grad_w[Z];
    auto [mu, zeta] = GetViscosity(rho);
    tau[XX] = 2 * mu * grad_u[X] + zeta * div_uvw;
    tau[YY] = 2 * mu * grad_v[Y] + zeta * div_uvw;
    tau[ZZ] = 2 * mu * grad_w[Z] + zeta * div_uvw;
    tau[XY] = mu * (grad_u[Y] + grad_v[X]);
    tau[YZ] = mu * (grad_v[Z] + grad_w[Y]);
    tau[ZX] = mu * (grad_w[X] + grad_u[Z]);
    return tau;
  }

 protected:
  static Scalar Dot(Scalar x,  Scalar y, Scalar z, Vector const &v) {
    return v[X] * x + v[Y] * y + v[Z] * z;
  }

 public:
  static void MinusViscousFlux(Conservative const &c_val, Gradient const &c_grad,
      FluxMatrix *flux) {
    auto [p_val, p_grad] = ConservativeToPrimitive(c_val, c_grad);
    Tensor tau = GetViscousStressTensor(p_grad, c_val.mass());
    Scalar kappa = GetThermalConductivity(c_val.mass());
    auto const &uvw = p_val.momentum();
    auto const &grad_T = p_grad.col(kTemperature);
    auto &&flux_x = flux->col(X);
    flux_x[U] -= tau[XX];
    flux_x[V] -= tau[XY];
    flux_x[W] -= tau[XZ];
    flux_x[kEnergy] -= Dot(tau[XX], tau[XY], tau[XZ], uvw) + kappa * grad_T[X];
    auto &&flux_y = flux->col(Y);
    flux_y[U] -= tau[YX];
    flux_y[V] -= tau[YY];
    flux_y[W] -= tau[YZ];
    flux_y[kEnergy] -= Dot(tau[YX], tau[YY], tau[YZ], uvw) + kappa * grad_T[Y];
    auto &&flux_z = flux->col(Z);
    flux_z[U] -= tau[ZX];
    flux_z[V] -= tau[ZY];
    flux_z[W] -= tau[ZZ];
    flux_z[kEnergy] -= Dot(tau[ZX], tau[ZY], tau[ZZ], uvw) + kappa * grad_T[Z];
  }

 protected:
  static void MinusViscousFlux(Scalar rho, Gradient const &p_grad,
      Vector const &normal, Vector const &uvw, Vector const &grad_T,
      Flux *flux) {
    Tensor tau = GetViscousStressTensor(p_grad, rho);
    auto &flux_momentum = flux->momentum();
    flux_momentum[X] -= Dot(tau[XX], tau[XY], tau[XZ], normal);
    flux_momentum[Y] -= Dot(tau[YX], tau[YY], tau[YZ], normal);
    flux_momentum[Z] -= Dot(tau[ZX], tau[ZY], tau[ZZ], normal);
    Scalar kappa = GetThermalConductivity(rho);
    auto work_x = Dot(tau[XX], tau[XY], tau[XZ], uvw) + kappa * grad_T[X];
    auto work_y = Dot(tau[YX], tau[YY], tau[YZ], uvw) + kappa * grad_T[Y];
    auto work_z = Dot(tau[ZX], tau[ZY], tau[ZZ], uvw) + kappa * grad_T[Z];
    flux->energy() -= Dot(work_x, work_y, work_z, normal);
  }

 public:
  static void MinusViscousFlux(Conservative const &c_val, Gradient const &c_grad,
      Vector const &normal, FluxVector *flux_vector) {
    auto [p_val, p_grad] = ConservativeToPrimitive(c_val, c_grad);
    auto *flux = static_cast<Flux *>(flux_vector);
    auto const &grad_T = p_grad.col(4);
    auto const &uvw = p_val.momentum();
    MinusViscousFlux(c_val.mass(), p_grad, normal, uvw, grad_T, flux);
  }

  static void MinusViscousFluxOnNoSlipWall(Value const &wall_value,
      Conservative const &c_val, Gradient const &c_grad,
      Vector const &normal, FluxVector *flux_vector) {
    auto [p_val, p_grad] = ConservativeToPrimitive(c_val, c_grad);
    auto *flux = static_cast<Flux *>(flux_vector);
    auto const &wall_value_ref = static_cast<Primitive const &>(wall_value);
    auto const &uvw = wall_value_ref.momentum();
    Vector grad_T = normal * wall_value_ref.energy();
    MinusViscousFlux(c_val.mass(), p_grad, normal, uvw, grad_T, flux);
  }

  static void SetValueOnNoSlipFace(Value const &wall_value, Value *value) {
    auto &primitive = static_cast<Primitive const &>(wall_value);
    auto *conservative = static_cast<Conservative *>(value);
    Scalar rho = conservative->mass();
    Scalar kinetic_old = conservative->momentum().squaredNorm() / rho;
    Scalar kinetic_new = primitive.momentum().squaredNorm() * rho;
    conservative->momentum() = rho * primitive.momentum();
    conservative->energy() += (kinetic_new - kinetic_old) / 2;
  }
};

template <typename G>
typename NavierStokes<G>::Scalar NavierStokes<G>::nu_;

template <typename G>
typename NavierStokes<G>::Scalar NavierStokes<G>::prandtl_;

}  // namespace diffusive
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_DIFFUSIVE_NAVIER_STOKES_HPP_
