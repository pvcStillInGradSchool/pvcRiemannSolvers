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
  struct Property {
    Scalar x, y, z;
    bool operator==(Property const &that) const {
      return x == that.x && y == that.y && z == that.z;
    }
  };

 protected:
  static Property nu_;

 public:
  static void SetProperty(Scalar nu_x, Scalar nu_y, Scalar nu_z) {
    nu_.x = nu_x; nu_.y = nu_y; nu_.z = nu_z;
  }

  static Property const &GetProperty() {
    return nu_;
  }

  template <class Int>
  static Property const &GetPropertyOnCell(Int i_cell, int i_node) {
    return GetProperty();
  }

  static void MinusViscousFlux(FluxMatrix *flux, Property const &nu,
      Conservative const &value, Gradient const &gradient) {
    using namespace mini::constant::index;
    flux->col(X) -= nu.x * gradient.row(X);
    flux->col(Y) -= nu.y * gradient.row(Y);
    flux->col(Z) -= nu.z * gradient.row(Z);
  }

  static void MinusViscousFlux(Flux *flux, Property const &nu,
      Conservative const &value, Gradient const &gradient,
      Vector const &normal) {
    using namespace mini::constant::index;
    *flux -= (normal[X] * nu.x) * gradient.row(X);
    *flux -= (normal[Y] * nu.y) * gradient.row(Y);
    *flux -= (normal[Z] * nu.z) * gradient.row(Z);
  }

  static void MinusViscousFluxOnNoSlipWall(Flux *flux,
      Property const &nu, Value const &wall_value,
      Conservative const &c_val, Gradient const &c_grad,
      Vector const &normal, Scalar value_penalty) {
    MinusViscousFlux(flux, nu, c_val, c_grad, normal);
  }
};

template <typename S, int K>
typename Anisotropic<S, K>::Property Anisotropic<S, K>::nu_;

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
  using Property = typename Base::Property;

 public:
  static void SetProperty(Scalar nu) {
    Base::SetProperty(nu, nu, nu);
  }
};

}  // namespace diffusive
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_DIFFUSIVE_SCALAR_HPP_
