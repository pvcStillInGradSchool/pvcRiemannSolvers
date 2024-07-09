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
  using Base = Simple<simple::Single<S, D>>;

 public:
  using Convection = Single;
  using Conservative = typename Base::Conservative;
  using Jacobian = typename Base::Jacobian;

  constexpr static int kComponents = 1;
  constexpr static int kDimensions = D;
  using Scalar = S;

 private:
  static Scalar max_eigen_value_;

 public:
  static void SetJacobians(Jacobian const &a_x, Jacobian const &a_y,
      Jacobian const &a_z) {
    Base::SetJacobians(a_x, a_y, a_z);
    static_assert(std::is_same_v<Jacobian, Scalar>);
    max_eigen_value_ = std::hypot(a_x, a_y, a_z);
  }

  static Scalar GetMaximumSpeed(Conservative const &conservative) {
    return max_eigen_value_;
  }

  static Scalar GetMaximumSpeedAndReferenceValueSquare(
        Conservative const &conservative, Conservative *squares) {
    squares->array() = conservative.array() * conservative.array();
    return GetMaximumSpeed(conservative);
  }
};

template <typename S, int D>
typename Single<S, D>::Scalar
Single<S, D>::max_eigen_value_;

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_ROTATED_SINGLE_HPP_
