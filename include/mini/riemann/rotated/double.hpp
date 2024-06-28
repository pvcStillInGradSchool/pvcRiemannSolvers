// Copyright 2019 PEI Weicheng and YANG Minghao
#ifndef MINI_RIEMANN_ROTATED_DOUBLE_HPP_
#define MINI_RIEMANN_ROTATED_DOUBLE_HPP_

#include <cmath>

#include "mini/riemann/rotated/simple.hpp"
#include "mini/riemann/simple/double.hpp"
#include "mini/geometry/pi.hpp"
#include "mini/algebra/eigen.hpp"

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

 private:
  static Scalar max_eigen_value_;

 public:
  static void SetJacobians(Jacobian const &a_x, Jacobian const &a_y,
      Jacobian const &a_z) {
    Base::SetJacobians(a_x, a_y, a_z);
    max_eigen_value_ = 0;
    for (Scalar theta = 0; theta < 180; theta += 10) {
      auto [cos_theta, sin_theta] = mini::geometry::CosSin(theta);
      for (Scalar phi = 0; phi < 360; phi += 10) {
        auto [cos_phi, sin_phi] = mini::geometry::CosSin(phi);
        Scalar n_x = sin_theta * cos_phi;
        Scalar n_y = sin_theta * sin_phi;
        Scalar n_z = cos_theta;
        Jacobian a_n = a_x * n_x + a_y * n_y + a_z * n_z;
        Conservative eig_vals = mini::algebra::GetEigenValues(a_n);
        max_eigen_value_ = std::max(max_eigen_value_, std::abs(eig_vals[0]));
        max_eigen_value_ = std::max(max_eigen_value_, std::abs(eig_vals[1]));
      }
    }
  }

  static Scalar GetMaximumSpeed(Conservative const &conservative) {
    return max_eigen_value_;
  }
};

template <typename S, int D>
typename Double<S, D>::Scalar
Double<S, D>::max_eigen_value_;

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_ROTATED_DOUBLE_HPP_
