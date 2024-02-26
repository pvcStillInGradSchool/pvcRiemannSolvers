// Copyright 2023 PEI Weicheng
#ifndef MINI_RIEMANN_DIFFUSIVE_SCALAR_HPP_
#define MINI_RIEMANN_DIFFUSIVE_SCALAR_HPP_

#include "mini/algebra/eigen.hpp"
#include "mini/constant/index.hpp"

namespace mini {
namespace riemann {
namespace diffusive {

/**
 * @brief A constant linear diffusion model, whose diffusive flux is \f$ \begin{bmatrix} \nu_x\,\partial_x\,u & \nu_y\,\partial_y\,u & \nu_z\,\partial_z\,u \end{bmatrix} \f$.
 * 
 * @tparam S 
 * @tparam K 
 */
template <typename S, int K>
class Anisotropic {
 public:
  static constexpr int kDimensions = 3;
  static constexpr int kComponents = K;
  using Scalar = S;
  using Vector = algebra::Vector<Scalar, kDimensions>;
  using Value = algebra::Vector<Scalar, kComponents>;
  using Conservative = Value;
  using Gradient = algebra::Matrix<Scalar, kDimensions, kComponents>;
  using FluxMatrix = algebra::Matrix<Scalar, kComponents, kDimensions>;
  using Flux = Conservative;

 private:
  static Scalar nu_x_;
  static Scalar nu_y_;
  static Scalar nu_z_;

 public:
  static void SetDiffusionCoefficient(Scalar nu_x, Scalar nu_y, Scalar nu_z) {
    nu_x_ = nu_x; nu_y_ = nu_y; nu_z_ = nu_z;
  }

  static void MinusViscousFlux(Conservative const &value, Gradient const &gradient,
      FluxMatrix *flux) {
    using namespace mini::constant::index;
    flux->col(X) -= nu_x_ * gradient.row(X);
    flux->col(Y) -= nu_y_ * gradient.row(Y);
    flux->col(Z) -= nu_z_ * gradient.row(Z);
  }

  static void MinusViscousFlux(Conservative const &value, Gradient const &gradient,
      Vector const &normal, Flux *flux) {
    using namespace mini::constant::index;
    *flux -= (normal[X] * nu_x_) * gradient.row(X);
    *flux -= (normal[Y] * nu_y_) * gradient.row(Y);
    *flux -= (normal[Z] * nu_z_) * gradient.row(Z);
  }

  static void MinusViscousFluxOnSlidingWall(Value const &wall_value,
      Conservative const &c_val, Gradient const &c_grad,
      Vector const &normal, Flux *flux) {
    MinusViscousFlux(c_val, c_grad, normal, flux);
  }
};
template <typename S, int K>
typename Anisotropic<S, K>::Scalar Anisotropic<S, K>::nu_x_;
template <typename S, int K>
typename Anisotropic<S, K>::Scalar Anisotropic<S, K>::nu_y_;
template <typename S, int K>
typename Anisotropic<S, K>::Scalar Anisotropic<S, K>::nu_z_;

/**
 * @brief A constant linear diffusion model, whose diffusive flux is \f$ \nu \begin{bmatrix} \partial_x\,u & \partial_y\,u & \partial_z\,u \end{bmatrix} \f$.
 * 
 * @tparam S
 * @tparam K 
 */
template <typename S, int K>
class Isotropic : public Anisotropic<S, K> {
  using Base = Anisotropic<S, K>;

 public:
  static constexpr int kDimensions = 3;
  static constexpr int kComponents = K;
  using Scalar = typename Base::Scalar;
  using Conservative = typename Base::Conservative;
  using Gradient = typename Base::Gradient;
  using FluxMatrix = typename Base::FluxMatrix;
  using Flux = typename Base::Flux;

 public:
  static void SetDiffusionCoefficient(Scalar nu) {
    Base::SetDiffusionCoefficient(nu, nu, nu);
  }
};

}  // namespace diffusive
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_DIFFUSIVE_SCALAR_HPP_
