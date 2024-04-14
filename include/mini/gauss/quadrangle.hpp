//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_GAUSS_QUADRANGLE_HPP_
#define MINI_GAUSS_QUADRANGLE_HPP_

#include <concepts>

#include <cmath>

#include <type_traits>

#include "mini/algebra/eigen.hpp"

#include "mini/gauss/line.hpp"
#include "mini/gauss/face.hpp"
#include "mini/geometry/quadrangle.hpp"

namespace mini {
namespace gauss {

/**
 * @brief Numerical integrators on quadrilateral elements.
 * 
 * @tparam kPhysDim  Dimension of the physical space.
 * @tparam Gx  The quadrature rule in the \f$\xi\f$ direction.
 * @tparam Gy  The quadrature rule in the \f$\eta\f$ direction.
 */
template <int kPhysDim, class Gx, class Gy>
class Quadrangle : public Face<typename Gx::Scalar, kPhysDim> {
  static constexpr int D = kPhysDim;

 public:
  using GaussX = Gx;
  using GaussY = Gy;
  using Scalar = typename GaussX::Scalar;
  static_assert(std::is_same_v<Scalar, typename Gy::Scalar>);
  using Lagrange = geometry::Quadrangle<Scalar, kPhysDim>;
  using Real = typename Lagrange::Real;
  using Local = typename Lagrange::Local;
  using Global = typename Lagrange::Global;
  using Jacobian = typename Lagrange::Jacobian;
  using Frame = typename Lagrange::Frame;

 private:
  using Base = Face<Scalar, kPhysDim>;
  static constexpr int Qx = GaussX::Q;
  static constexpr int Qy = GaussY::Q;
  static constexpr int Q = Qx * Qy;
  static const std::array<Local, Q> local_coords_;
  static const std::array<Scalar, Q> local_weights_;
  std::array<Global, Qx * Qy> global_coords_;
  std::array<Scalar, Qx * Qy> global_weights_;
  std::array<Frame, Qx * Qy> normal_frames_;
  Lagrange const *lagrange_;
  Scalar area_;

 public:
  int CountPoints() const final {
    return Qx * Qy;
  }

 private:
  static constexpr auto BuildLocalCoords() {
    std::array<Local, Qx * Qy> points;
    int k = 0;
    for (int i = 0; i < Qx; ++i) {
      for (int j = 0; j < Qy; ++j) {
        points[k][0] = GaussX::points[i];
        points[k][1] = GaussY::points[j];
        k++;
      }
    }
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, Qx * Qy> weights;
    int k = 0;
    for (int i = 0; i < Qx; ++i) {
      for (int j = 0; j < Qy; ++j) {
        weights[k++] = GaussX::weights[i] * GaussY::weights[j];
      }
    }
    return weights;
  }

 public:
  const Global &GetGlobalCoord(int i) const final {
    assert(0 <= i && i < CountPoints());
    return global_coords_[i];
  }
  const Scalar &GetGlobalWeight(int i) const final {
    assert(0 <= i && i < CountPoints());
    return global_weights_[i];
  }
  const Local &GetLocalCoord(int i) const final {
    assert(0 <= i && i < CountPoints());
    return local_coords_[i];
  }
  const Scalar &GetLocalWeight(int i) const final {
    assert(0 <= i && i < CountPoints());
    return local_weights_[i];
  }

 protected:
  Global &GetGlobalCoord(int i) final {
    assert(0 <= i && i < CountPoints());
    return global_coords_[i];
  }
  Scalar &GetGlobalWeight(int i) final {
    assert(0 <= i && i < CountPoints());
    return global_weights_[i];
  }

 public:
  const Frame &GetNormalFrame(int i) const final {
    assert(0 <= i && i < CountPoints());
    return normal_frames_[i];
  }
  Frame &GetNormalFrame(int i) final {
    assert(0 <= i && i < CountPoints());
    return normal_frames_[i];
  }

 public:
  explicit Quadrangle(Lagrange const &lagrange)
      : lagrange_(&lagrange) {
    area_ = this->BuildQuadraturePoints();
    Base::BuildNormalFrames(this);
  }

  const Lagrange &lagrange() const final {
    return *lagrange_;
  }

  Scalar area() const final {
    return area_;
  }
};

template <int D, class Gx, class Gy>
std::array<typename Quadrangle<D, Gx, Gy>::Local,
    Quadrangle<D, Gx, Gy>::Q> const
Quadrangle<D, Gx, Gy>::local_coords_
    = Quadrangle<D, Gx, Gy>::BuildLocalCoords();

template <int D, class Gx, class Gy>
std::array<typename Quadrangle<D, Gx, Gy>::Scalar,
    Quadrangle<D, Gx, Gy>::Q> const
Quadrangle<D, Gx, Gy>::local_weights_
    = Quadrangle<D, Gx, Gy>::BuildLocalWeights();

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_QUADRANGLE_HPP_
