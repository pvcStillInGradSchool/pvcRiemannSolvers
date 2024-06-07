// Copyright 2019 PEI Weicheng and YANG Minghao
#ifndef MINI_RIEMANN_ROTATED_BURGERS_HPP_
#define MINI_RIEMANN_ROTATED_BURGERS_HPP_

#include "mini/riemann/rotated/simple.hpp"
#include "mini/riemann/simple/burgers.hpp"

namespace mini {
namespace riemann {
namespace rotated {

template <typename S, int D>
class Burgers : public Simple<simple::Burgers<S, D>> {
  using Base = Simple<simple::Burgers<S, D>>;

 public:
  using Convection = Burgers;

  constexpr static int kComponents = 1;
  constexpr static int kDimensions = D;
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using Conservative = typename Base::Conservative;
  using Flux = typename Base::Flux;
  using FluxMatrix = typename Base::FluxMatrix;

  static FluxMatrix GetFluxMatrix(const Conservative& state) {
    FluxMatrix flux_mat;
    for (int c = 0; c < D; ++c) {
      flux_mat(0, c) = state[0] * state[0];
      flux_mat(0, c) *= Base::convection_coefficient_[c] / 2.0;
    }
    return flux_mat;
  }
};

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_ROTATED_BURGERS_HPP_
