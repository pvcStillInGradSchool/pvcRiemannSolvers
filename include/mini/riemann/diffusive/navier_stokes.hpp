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
  using FluxVector = algebra::Vector<Scalar, kComponents>;
  using Conservative = euler::Conservatives<Scalar, kDimensions>;
  using Primitive = euler::Primitives<Scalar, kDimensions>;
  using Vector = typename Primitive::Vector;
  using Tensor = algebra::Vector<Scalar, 6>;

  static void SetProperty(Scalar mu, Scalar kappa) {
    mu_ = mu;
    zeta_ = -2.0 / 3 * mu;  // Stokes' hypothesis
    kappa_ = kappa;
  }

 protected:
  static Scalar mu_;
  static Scalar zeta_;
  static Scalar kappa_;

  static std::pair<Primitive, Gradient> ConservativeToPrimitive(
      Conservative const &c_val, Gradient const &c_grad) {
    auto p_val = Gas::ConservativeToPrimitive(c_val);
    Gradient p_grad;
    auto &&grad_u = p_grad.col(X);
    auto &&grad_v = p_grad.col(Y);
    auto &&grad_w = p_grad.col(Z);
    auto &grad_rho = c_grad.col(0);
    auto &grad_rho_u = c_grad.col(1 + X);
    auto &grad_rho_v = c_grad.col(1 + Y);
    auto &grad_rho_w = c_grad.col(1 + Z);
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
    auto &&grad_p = p_grad.col(3);
    grad_p  = u * grad_rho_u + rho_u * grad_u;
    grad_p += v * grad_rho_v + rho_v * grad_v;
    grad_p += w * grad_rho_w + rho_w * grad_w;
    grad_p *= -0.5;
    grad_p += c_grad.col(4);
    grad_p *= Gas::GammaMinusOne();
    auto &&grad_T = p_grad.col(4);
    grad_T = grad_p / rho - (p / (rho * rho)) * grad_rho;
    grad_T /= Gas::R();
    return {p_val, p_grad};
  }

  static Tensor GetViscousStressTensor(Gradient const &p_grad) {
    Tensor tau;
    const auto &grad_u = p_grad.col(X);
    const auto &grad_v = p_grad.col(Y);
    const auto &grad_w = p_grad.col(Z);
    auto div_uvw = grad_u[X] + grad_v[Y] + grad_w[Z];
    tau[XX] = 2 * mu_ * grad_u[X] + zeta_ * div_uvw;
    tau[YY] = 2 * mu_ * grad_v[Y] + zeta_ * div_uvw;
    tau[ZZ] = 2 * mu_ * grad_w[Z] + zeta_ * div_uvw;
    tau[XY] = mu_ * (grad_u[Y] + grad_v[X]);
    tau[YZ] = mu_ * (grad_v[Z] + grad_w[Y]);
    tau[ZX] = mu_ * (grad_w[X] + grad_u[Z]);
    return tau;
  }

  static Scalar Dot(Scalar x,  Scalar y, Scalar z, Vector const &v) {
    return v[X] * x + v[Y] * y + v[Z] * z;
  }

 public:
  static void MinusViscousFlux(Conservative const &c_val, Gradient const &c_grad,
      FluxMatrix *flux) {
    auto [p_val, p_grad] = ConservativeToPrimitive(c_val, c_grad);
    Tensor tau = GetViscousStressTensor(p_grad);
    auto const &uvw = p_val.momentum();
    auto const &grad_T = p_grad.col(4);
    auto &&flux_x = flux->col(X);
    flux_x[1] -= tau[XX];
    flux_x[2] -= tau[XY];
    flux_x[3] -= tau[XZ];
    flux_x[4] -= Dot(tau[XX], tau[XY], tau[XZ], uvw) + kappa_ * grad_T[X];
    auto &&flux_y = flux->col(Y);
    flux_y[1] -= tau[YX];
    flux_y[2] -= tau[YY];
    flux_y[3] -= tau[YZ];
    flux_y[4] -= Dot(tau[YX], tau[YY], tau[YZ], uvw) + kappa_ * grad_T[Y];
    auto &&flux_z = flux->col(Z);
    flux_z[1] -= tau[ZX];
    flux_z[2] -= tau[ZY];
    flux_z[3] -= tau[ZZ];
    flux_z[4] -= Dot(tau[ZX], tau[ZY], tau[ZZ], uvw) + kappa_ * grad_T[Z];
  }

  static void MinusViscousFlux(Conservative const &c_val, Gradient const &c_grad,
      Vector const &normal, FluxVector *flux_vector) {
    auto [p_val, p_grad] = ConservativeToPrimitive(c_val, c_grad);
    Tensor tau = GetViscousStressTensor(p_grad);
    auto *flux = static_cast<Flux *>(flux_vector);
    auto &flux_momentum = flux->momentum();
    flux_momentum[X] -= Dot(tau[XX], tau[XY], tau[XZ], normal);
    flux_momentum[Y] -= Dot(tau[YX], tau[YY], tau[YZ], normal);
    flux_momentum[Z] -= Dot(tau[ZX], tau[ZY], tau[ZZ], normal);
    auto const &grad_T = p_grad.col(4);
    auto const &uvw = p_val.momentum();
    auto work_x = Dot(tau[XX], tau[XY], tau[XZ], uvw) + kappa_ * grad_T[X];
    auto work_y = Dot(tau[YX], tau[YY], tau[YZ], uvw) + kappa_ * grad_T[Y];
    auto work_z = Dot(tau[ZX], tau[ZY], tau[ZZ], uvw) + kappa_ * grad_T[Z];
    flux->energy() -= Dot(work_x, work_y, work_z, normal);
  }
};

template <typename G>
typename NavierStokes<G>::Scalar NavierStokes<G>::mu_;

template <typename G>
typename NavierStokes<G>::Scalar NavierStokes<G>::zeta_;

template <typename G>
typename NavierStokes<G>::Scalar NavierStokes<G>::kappa_;

}  // namespace diffusive
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_DIFFUSIVE_NAVIER_STOKES_HPP_
