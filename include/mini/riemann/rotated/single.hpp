// Copyright 2019 PEI Weicheng and YANG Minghao
#ifndef MINI_RIEMANN_ROTATED_SINGLE_HPP_
#define MINI_RIEMANN_ROTATED_SINGLE_HPP_

#include "mini/riemann/rotated/simple.hpp"
#include "mini/riemann/simple/single.hpp"

namespace mini {
namespace riemann {
namespace rotated {

template <typename S, int D>
class Single : public Simple<simple::Single<S, D>> {
 public:
  using Convection = Single;

  constexpr static int kComponents = 1;
  constexpr static int kDimensions = D;
  using Scalar = S;
};

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_ROTATED_SINGLE_HPP_
