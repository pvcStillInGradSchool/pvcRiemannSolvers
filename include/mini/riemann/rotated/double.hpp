// Copyright 2019 PEI Weicheng and YANG Minghao
#ifndef MINI_RIEMANN_ROTATED_DOUBLE_HPP_
#define MINI_RIEMANN_ROTATED_DOUBLE_HPP_

#include "mini/riemann/rotated/simple.hpp"
#include "mini/riemann/simple/double.hpp"

namespace mini {
namespace riemann {
namespace rotated {

template <typename S, int D>
class Double : public Simple<simple::Double<S, D>> {
  using Base = Simple<simple::Double<S, D>>;

 public:
  using Convection = Double;

  constexpr static int kComponents = 2;
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

#endif  // MINI_RIEMANN_ROTATED_DOUBLE_HPP_
