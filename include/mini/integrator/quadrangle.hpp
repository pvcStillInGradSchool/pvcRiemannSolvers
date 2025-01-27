//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_INTEGRATOR_QUADRANGLE_HPP_
#define MINI_INTEGRATOR_QUADRANGLE_HPP_

#include <concepts>

#include <cmath>
#include <memory>
#include <type_traits>

#include "mini/algebra/eigen.hpp"

#include "mini/integrator/face.hpp"
#include "mini/coordinate/face.hpp"
#include "mini/coordinate/quadrangle.hpp"

namespace mini {
namespace integrator {

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
  using IntegratorX = Gx;
  using IntegratorY = Gy;
  using Scalar = typename IntegratorX::Scalar;
  static_assert(std::is_same_v<Scalar, typename Gy::Scalar>);
  using Coordinate = coordinate::Quadrangle<Scalar, kPhysDim>;
  using Real = typename Coordinate::Real;
  using Local = typename Coordinate::Local;
  using Global = typename Coordinate::Global;
  using Jacobian = typename Coordinate::Jacobian;
  using Frame = typename Coordinate::Frame;
  using Base = Face<Scalar, kPhysDim>;

 private:
  static constexpr int Qx = IntegratorX::Q;
  static constexpr int Qy = IntegratorY::Q;
  static constexpr int Q = Qx * Qy;
  static const std::array<Local, Q> local_coords_;
  static const std::array<Scalar, Q> local_weights_;
  std::array<Global, Qx * Qy> global_coords_;
  std::array<Scalar, Qx * Qy> global_weights_;
  std::array<Frame, Qx * Qy> normal_frames_;
  Coordinate const *coordinate_;
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
        points[k][0] = IntegratorX::points[i];
        points[k][1] = IntegratorY::points[j];
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
        weights[k++] = IntegratorX::weights[i] * IntegratorY::weights[j];
      }
    }
    return weights;
  }

 public:
  const Global &GetGlobal(int i) const final {
    assert(0 <= i && i < CountPoints());
    return global_coords_[i];
  }
  const Scalar &GetGlobalWeight(int i) const final {
    assert(0 <= i && i < CountPoints());
    return global_weights_[i];
  }
  const Local &GetLocal(int i) const final {
    assert(0 <= i && i < CountPoints());
    return local_coords_[i];
  }
  const Scalar &GetLocalWeight(int i) const final {
    assert(0 <= i && i < CountPoints());
    return local_weights_[i];
  }

 protected:
  Global &GetGlobal(int i) final {
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

 protected:
  Frame &GetNormalFrame(int i) final {
    assert(0 <= i && i < CountPoints());
    return normal_frames_[i];
  }

 public:
  std::unique_ptr<Base>
  Clone(typename Coordinate::Base const &coordinate) const final {
    return std::make_unique<Quadrangle>(
        dynamic_cast<Coordinate const &>(coordinate));
  }

  explicit Quadrangle(Coordinate const &lagrange)
      : coordinate_(&lagrange) {
    area_ = this->BuildQuadraturePoints();
    Base::BuildNormalFrames(this);
  }

  const Coordinate &coordinate() const final {
    return *coordinate_;
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

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_QUADRANGLE_HPP_
