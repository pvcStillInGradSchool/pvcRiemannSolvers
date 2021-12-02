// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_RIEMANN_ROTATED_SIMPLE_HPP_
#define MINI_RIEMANN_ROTATED_SIMPLE_HPP_

#include <cstring>

#include "mini/algebra/column.hpp"
#include "mini/algebra/eigen.hpp"

namespace mini {
namespace riemann {
namespace rotated {

template <class UnrotatedSimple>
class Simple {
 protected:
  using Base = UnrotatedSimple;
  static constexpr int K = Base::kFunc;
  static constexpr int D = Base::kDim;
  static constexpr int x{0}, y{1}, z{2};

 public:
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using MatKx1 = algebra::Matrix<Scalar, K, 1>;
  using State = MatKx1;
  using Flux = MatKx1;
  using FluxMatrix = algebra::Matrix<Scalar, K, D>;
  using Frame3d = mini::algebra::Matrix<Scalar, 3, 3>;
  using Jacobi = typename Base::Jacobi;
  using Coefficient = typename Base::Coefficient;

 protected:
  template <class Value>
  static Flux ConvertToFlux(const Value& v) {
    Flux flux;
    std::memcpy(flux.data(), &v, K * sizeof(flux[0]));
    return flux;
  }

 public:
  void Rotate(const Scalar& n_x, const Scalar& n_y) {
    static_assert(D == 2);
    auto a_normal = global_coefficient[x] * n_x;
    a_normal += global_coefficient[y] * n_y;
    unrotated_simple_ = UnrotatedSimple(a_normal);
  }
  void Rotate(const Scalar& n_x, const Scalar& n_y,  const Scalar& n_z) {
    static_assert(D == 3);
    auto a_normal = global_coefficient[x] * n_x;
    a_normal += global_coefficient[y] * n_y;
    a_normal += global_coefficient[z] * n_z;
    unrotated_simple_ = UnrotatedSimple(a_normal);
  }
  void Rotate(const Frame3d &frame) {
    const auto& nu = frame.col(0);
    Rotate(nu[x], nu[y], nu[z]);
  }
  Flux GetFluxOnTimeAxis(const State& left, const State& right) {
    auto raw_flux = unrotated_simple_.GetFluxOnTimeAxis(left, right);
    return ConvertToFlux(raw_flux);
  }
  Flux GetFluxOnSolidWall(const State& state) {
    Flux flux;
    flux.setZero();
    return flux;
  }
  Flux GetFluxOnFreeWall(const State& state) {
    auto raw_flux = unrotated_simple_.GetFlux(state);
    return ConvertToFlux(raw_flux);
  }
  Flux GetRotatedFlux(const State& state) {
    auto raw_flux = unrotated_simple_.GetFlux(state);
    return ConvertToFlux(raw_flux);
  }
  static FluxMatrix GetFluxMatrix(const State& state) {
    FluxMatrix flux_mat;
    for (int c = 0; c < D; ++c) {
      flux_mat.col(c) = state;
      flux_mat.col(c) *= global_coefficient[c];
    }
    return flux_mat;
  }

  static Coefficient global_coefficient;

 protected:
  UnrotatedSimple unrotated_simple_;
};
template <class UnrotatedSimple>
typename Simple<UnrotatedSimple>::Coefficient
Simple<UnrotatedSimple>::global_coefficient;

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_ROTATED_SIMPLE_HPP_
