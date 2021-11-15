// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_RIEMANN_ROTATED_SINGLE_HPP_
#define MINI_RIEMANN_ROTATED_SINGLE_HPP_

#include "mini/riemann/rotated/simple.hpp"
#include "mini/riemann/linear/single.hpp"

namespace mini {
namespace riemann {
namespace rotated {

template <int kDim = 2>
class Single : public Simple<linear::Single<kDim>> {
};

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_ROTATED_SINGLE_HPP_
