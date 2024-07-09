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

 private:
  static Scalar max_eigen_value_;

 public:
  static void SetJacobians(Jacobian const &a_x, Jacobian const &a_y,
      Jacobian const &a_z) {
    Base::SetJacobians(a_x, a_y, a_z);
    Conservative eigen_values;
    // TODO(PVC): generalize to non-diagonal Jacobian's
    for (int k = 0; k < K; ++k) {
      eigen_values[k] = std::hypot(a_x(k, k), a_y(k, k), a_z(k, k));
    }
    max_eigen_value_ = eigen_values.maxCoeff();
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

template <typename S, int K, int D>
typename Multiple<S, K, D>::Scalar
Multiple<S, K, D>::max_eigen_value_;

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_ROTATED_MULTIPLE_HPP_
