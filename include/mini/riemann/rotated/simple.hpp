// Copyright 2019 PEI Weicheng and YANG Minghao
#ifndef MINI_RIEMANN_ROTATED_SIMPLE_HPP_
#define MINI_RIEMANN_ROTATED_SIMPLE_HPP_

#include <cstring>

#include "mini/algebra/eigen.hpp"
#include "mini/constant/index.hpp"

namespace mini {
namespace riemann {
namespace rotated {

using namespace mini::constant::index;

template <class UnrotatedSimple>
class Simple {
 protected:
  using Base = UnrotatedSimple;
  static constexpr int K = Base::kComponents;
  static constexpr int D = Base::kDimensions;

 public:
  static constexpr int kComponents = Base::kComponents;
  static constexpr int kDimensions = D;
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using MatKx1 = algebra::Matrix<Scalar, K, 1>;
  using Conservative = MatKx1;
  using Flux = MatKx1;
  using FluxMatrix = algebra::Matrix<Scalar, K, D>;
  using Frame = std::array<Vector, kDimensions>;
  using Jacobian = typename Base::Jacobian;
  using Coefficient = typename Base::Coefficient;

 protected:
  template <class Value>
  static Flux ConvertToFlux(const Value& v) {
    Flux flux;
    std::memcpy(flux.data(), &v, K * sizeof(flux[0]));
    return flux;
  }

 public:
  void Rotate(const Frame &frame) {
    frame_ = &frame;
    const auto &nu = frame[X];
    assert(std::abs(1 - nu.norm()) < 1e-6);
    Jacobian a_normal = convection_coefficient_[X] * nu[X];
    a_normal += convection_coefficient_[Y] * nu[Y];
    a_normal += convection_coefficient_[Z] * nu[Z];
    unrotated_simple_ = UnrotatedSimple(a_normal);
  }
  Vector const &normal() const {
    return (*frame_)[X];
  }
  Flux GetFluxUpwind(const Conservative& left,
      const Conservative& right) const {
    auto raw_flux = unrotated_simple_.GetFluxUpwind(left, right);
    return ConvertToFlux(raw_flux);
  }
  Flux GetFluxOnInviscidWall(const Conservative& state) const {
    Flux flux;
    flux.setZero();
    return flux;
  }
  Flux GetFluxOnSupersonicOutlet(const Conservative& state) const {
    auto raw_flux = unrotated_simple_.GetFlux(state);
    return ConvertToFlux(raw_flux);
  }
  Flux GetFluxOnSupersonicInlet(const Conservative& state) const {
    auto raw_flux = unrotated_simple_.GetFlux(state);
    return ConvertToFlux(raw_flux);
  }
  Flux GetFluxOnSubsonicInlet(Conservative const& conservative_i,
      Conservative const& conservative_o) const {
    return {};
  }
  Flux GetFluxOnSubsonicOutlet(Conservative const& conservative_i,
      Conservative const& conservative_o) const {
    return {};
  }
  Flux GetFluxOnSmartBoundary(Conservative const& conservative_i,
      Conservative const& conservative_o) const {
    // TODO(PVC): provide physical implementation
    return {};
  }
  static FluxMatrix GetFluxMatrix(const Conservative& state) {
    FluxMatrix flux_mat;
    flux_mat.col(X) = convection_coefficient_[X] * state;
    flux_mat.col(Y) = convection_coefficient_[Y] * state;
    flux_mat.col(Z) = convection_coefficient_[Z] * state;
    return flux_mat;
  }
  static void SetJacobians(Jacobian const &a_x, Jacobian const &a_y,
      Jacobian const &a_z) {
    convection_coefficient_[X] = a_x;
    convection_coefficient_[Y] = a_y;
    convection_coefficient_[Z] = a_z;
  }

 protected:
  UnrotatedSimple unrotated_simple_;
  const Frame *frame_;
  static Coefficient convection_coefficient_;
};
template <class UnrotatedSimple>
typename Simple<UnrotatedSimple>::Coefficient
Simple<UnrotatedSimple>::convection_coefficient_;

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_ROTATED_SIMPLE_HPP_
