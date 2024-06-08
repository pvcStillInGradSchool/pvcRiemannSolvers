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
  struct Property {
    Scalar nu, prandtl;
  };

 protected:
  static Property property_;

 public:
  static void SetProperty(Scalar nu, Scalar prandtl) {
    property_.nu = nu;
    property_.prandtl = prandtl;
  }
  static Property const &GetProperty() {
    return property_;
  }

  template <class Int>
  static Property const &GetPropertyOnCell(Int i_cell, int i_node) {
    return GetProperty();
  }

  static std::pair<Scalar, Scalar> GetViscosity(Scalar rho, Scalar nu) {
    Scalar mu = nu * rho;
    Scalar zeta = -2.0 / 3 * mu;  // Stokes' hypothesis
    return {mu, zeta};
  }

  static Scalar GetThermalConductivity(Scalar rho, Property const &property) {
    Scalar mu = property.nu * rho;
    return mu * Gas::Cp() / property.prandtl;
  }

 protected:
  static constexpr int kMass = 0;
  static constexpr int U = 1 + X;
  static constexpr int V = 1 + Y;
  static constexpr int W = 1 + Z;
  static constexpr int kEnergy = 4;
  static constexpr int kPressure = 4;
  static constexpr int kTemperature = 4;

 public:
  static std::pair<Primitive, Gradient> ConservativeToPrimitive(
      Conservative const &conservative, Gradient const &grad_conservative) {
    auto primitive = Gas::ConservativeToPrimitive(conservative);
    Gradient grad_primitive;
    auto &&grad_u = grad_primitive.col(U);
    auto &&grad_v = grad_primitive.col(V);
    auto &&grad_w = grad_primitive.col(W);
    auto &grad_rho = grad_conservative.col(kMass);
    auto &grad_rho_u = grad_conservative.col(U);
    auto &grad_rho_v = grad_conservative.col(V);
    auto &grad_rho_w = grad_conservative.col(W);
    grad_primitive.col(kMass) = grad_rho;
    auto rho = primitive.rho();
    auto u = primitive.u();
    auto v = primitive.v();
    auto w = primitive.w();
    auto p = primitive.p();
    grad_u = (grad_rho_u - u * grad_rho) / rho;
    grad_v = (grad_rho_v - v * grad_rho) / rho;
    grad_w = (grad_rho_w - w * grad_rho) / rho;
    auto rho_u = conservative.momentumX();
    auto rho_v = conservative.momentumY();
    auto rho_w = conservative.momentumZ();
    auto &&grad_p = grad_primitive.col(kPressure);
    grad_p  = u * grad_rho_u + rho_u * grad_u;
    grad_p += v * grad_rho_v + rho_v * grad_v;
    grad_p += w * grad_rho_w + rho_w * grad_w;
    grad_p *= -0.5;
    grad_p += grad_conservative.col(kEnergy);
    grad_p *= Gas::GammaMinusOne();
    auto &&grad_T = grad_primitive.col(kTemperature);
    grad_T = grad_p / rho - (p / (rho * rho)) * grad_rho;
    grad_T /= Gas::R();
    return {primitive, grad_primitive};
  }

  static Tensor GetViscousStressTensor(Gradient const &grad_primitive,
      Scalar rho, Scalar nu) {
    Tensor tau;
    const auto &grad_u = grad_primitive.col(U);
    const auto &grad_v = grad_primitive.col(V);
    const auto &grad_w = grad_primitive.col(W);
    auto div_uvw = grad_u[X] + grad_v[Y] + grad_w[Z];
    auto [mu, zeta] = GetViscosity(rho, nu);
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
  static void MinusViscousFlux(FluxMatrix *flux, Property const &property,
      Conservative const &conservative, Gradient const &grad_conservative) {
    auto [primitive, grad_primitive]
        = ConservativeToPrimitive(conservative, grad_conservative);
    Tensor tau = GetViscousStressTensor(grad_primitive, conservative.mass(),
        property.nu);
    Scalar kappa = GetThermalConductivity(conservative.mass(), property);
    auto const &uvw = primitive.velocity();
    auto const &grad_T = grad_primitive.col(kTemperature);
    flux->row(kEnergy) -= kappa * grad_T;
    auto &&flux_x = flux->col(X);
    flux_x[U] -= tau[XX];
    flux_x[V] -= tau[XY];
    flux_x[W] -= tau[XZ];
    flux_x[kEnergy] -= Dot(tau[XX], tau[XY], tau[XZ], uvw);
    auto &&flux_y = flux->col(Y);
    flux_y[U] -= tau[YX];
    flux_y[V] -= tau[YY];
    flux_y[W] -= tau[YZ];
    flux_y[kEnergy] -= Dot(tau[YX], tau[YY], tau[YZ], uvw);
    auto &&flux_z = flux->col(Z);
    flux_z[U] -= tau[ZX];
    flux_z[V] -= tau[ZY];
    flux_z[W] -= tau[ZZ];
    flux_z[kEnergy] -= Dot(tau[ZX], tau[ZY], tau[ZZ], uvw);
  }

 protected:
  static void _MinusViscousFlux(Flux *flux, Property const &property,
      Scalar rho, Gradient const &grad_primitive,
      Vector const &normal, Vector const &uvw, Scalar normal_dot_grad_T) {
    Tensor tau = GetViscousStressTensor(grad_primitive, rho, property.nu);
    flux->momentumX() -= Dot(tau[XX], tau[XY], tau[XZ], normal);
    flux->momentumY() -= Dot(tau[YX], tau[YY], tau[YZ], normal);
    flux->momentumZ() -= Dot(tau[ZX], tau[ZY], tau[ZZ], normal);
    Scalar kappa = GetThermalConductivity(rho, property);
    Scalar work_x = Dot(tau[XX], tau[XY], tau[XZ], uvw);
    Scalar work_y = Dot(tau[YX], tau[YY], tau[YZ], uvw);
    Scalar work_z = Dot(tau[ZX], tau[ZY], tau[ZZ], uvw);
    flux->energy() -= Dot(work_x, work_y, work_z, normal)
        + kappa * normal_dot_grad_T;
  }

 public:
  static void MinusViscousFlux(Conservative const &conservative,
      Gradient const &grad_conservative, Vector const &normal,
      FluxVector *flux_vector) {
    MinusViscousFlux(flux_vector, property_, conservative, grad_conservative, normal);
  }
  static void MinusViscousFlux(FluxVector *flux_vector, Property const &property,
      Conservative const &conservative, Gradient const &grad_conservative, Vector const &normal) {
    auto [primitive, grad_primitive]
        = ConservativeToPrimitive(conservative, grad_conservative);
    auto *flux = static_cast<Flux *>(flux_vector);
    auto const &uvw = primitive.velocity();
    auto const &grad_T = grad_primitive.col(kTemperature);
    _MinusViscousFlux(flux, property, conservative.mass(), grad_primitive, normal, uvw,
        normal.dot(grad_T));
  }

  static void MinusViscousFluxOnNoSlipWall(Value const &wall_value,
      Conservative const &conservative,
      Gradient const &grad_conservative,
      Vector const &normal, Scalar value_penalty,
      FluxVector *flux_vector) {
    auto [primitive, grad_primitive]
        = ConservativeToPrimitive(conservative, grad_conservative);
    auto *flux = static_cast<Flux *>(flux_vector);
    auto const &wall_value_ref = static_cast<Primitive const &>(wall_value);
    auto const &uvw = wall_value_ref.velocity();
    Vector penalty = value_penalty * (uvw - primitive.velocity());
    grad_primitive(X, U) += normal[X] * penalty[X];
    grad_primitive(X, V) += normal[X] * penalty[Y];
    grad_primitive(X, W) += normal[X] * penalty[Z];
    grad_primitive(Y, U) += normal[Y] * penalty[X];
    grad_primitive(Y, V) += normal[Y] * penalty[Y];
    grad_primitive(Y, W) += normal[Y] * penalty[Z];
    grad_primitive(Z, U) += normal[Z] * penalty[X];
    grad_primitive(Z, V) += normal[Z] * penalty[Y];
    grad_primitive(Z, W) += normal[Z] * penalty[Z];
    _MinusViscousFlux(flux, property_, conservative.mass(), grad_primitive, normal, uvw,
        wall_value_ref.energy());
  }
};

template <typename G>
typename NavierStokes<G>::Property NavierStokes<G>::property_;

}  // namespace diffusive
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_DIFFUSIVE_NAVIER_STOKES_HPP_
