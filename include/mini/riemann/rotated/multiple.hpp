// Copyright 2019 PEI Weicheng and YANG Minghao
#ifndef MINI_RIEMANN_ROTATED_MULTIPLE_HPP_
#define MINI_RIEMANN_ROTATED_MULTIPLE_HPP_

#include "mini/riemann/rotated/simple.hpp"
#include "mini/riemann/simple/multiple.hpp"

namespace mini {
namespace riemann {
namespace rotated {

template <typename S, int K, int D>
class Multiple : public Simple<simple::Multiple<S, K, D>> {
  using Base = Simple<simple::Multiple<S, K, D>>;

 public:
  using Convection = Multiple;

  constexpr static int kComponents = K;
  constexpr static int kDimensions = D;
  using Scalar = S;
  using Jacobian = typename Base::Jacobian;
  using Conservative = typename Base::Conservative;

  void UpdateEigenMatrices(const Conservative &) {
  }
  const Jacobian& L() const {
    return this->unrotated_simple_.L();
  }
  const Jacobian& R() const {
    return this->unrotated_simple_.R();
  }
};

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_ROTATED_MULTIPLE_HPP_
