//  Copyright 2023 PEI Weicheng
#ifndef MINI_INTEGRATOR_PYRAMID_HPP_
#define MINI_INTEGRATOR_PYRAMID_HPP_

#include <concepts>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <type_traits>

#include "mini/integrator/jacobi.hpp"
#include "mini/integrator/cell.hpp"
#include "mini/coordinate/pyramid.hpp"

namespace mini {
namespace integrator {
/**
 * @brief Numerical integrators on pyramidal elements.
 * 
 * @tparam Gx  The quadrature rule in the \f$\xi\f$ direction.
 * @tparam Gy  The quadrature rule in the \f$\eta\f$ direction.
 * @tparam Qz  Number of qudrature points in the \f$\zeta\f$ direction.
 */
template <class Gx, class Gy, int Qz>
class Pyramid : public Cell<typename Gx::Scalar> {
 public:
  using Scalar = typename Gx::Scalar;
  using Base = Cell<Scalar>;
  using IntegratorX = Gx;
  using IntegratorY = Gy;
  using IntegratorZ = Jacobi<Scalar, Qz, 2, 0>;
  using Coordinate = coordinate::Pyramid<Scalar>;
  using Real = typename Coordinate::Real;
  using Local = typename Coordinate::Local;
  using Global = typename Coordinate::Global;
  static_assert(std::is_same_v<Local, Global>);
  using Jacobian = typename Coordinate::Jacobian;

 private:
  static constexpr int Qx = IntegratorX::Q;
  static constexpr int Qy = IntegratorY::Q;

 public:
  static constexpr int Q = Qx * Qy * Qz;

 private:
  using Points = std::array<Local, Qx * Qy * Qz>;
  static const Points local_coords_;

  using Weights = std::array<Scalar, Qx * Qy * Qz>;
  static const Weights local_weights_;

  Points global_coords_;
  Weights global_weights_;
  Coordinate const *coordinate_;
  Scalar volume_;

 public:
  int CountPoints() const final {
    return Q;
  }

 private:
  static constexpr auto BuildLocalCoords() {
    Points points;
    int n = 0;
    for (int i = 0; i < Qx; ++i) {
      for (int j = 0; j < Qy; ++j) {
        for (int k = 0; k < Qz; ++k) {
          points[n][X] = IntegratorX::points[i];
          points[n][Y] = IntegratorY::points[j];
          points[n][Z] = IntegratorZ::points[k];
          n++;
        }
      }
    }
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    Weights weights;
    int n = 0;
    for (int i = 0; i < Qx; ++i) {
      for (int j = 0; j < Qy; ++j) {
        for (int k = 0; k < Qz; ++k) {
          weights[n++] = IntegratorX::weights[i] * IntegratorY::weights[j]
              // Jacobi::weights have taken account of \f$ (1 - \xi)^2 \f$.
              * IntegratorZ::weights[k] * std::pow(1 - IntegratorZ::points[k], -2);
        }
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
  std::unique_ptr<Base>
  Clone(typename Coordinate::Base const &coordinate) const final {
    return std::make_unique<Pyramid>(
        dynamic_cast<Coordinate const &>(coordinate));
  }

  explicit Pyramid(Coordinate const &lagrange)
      : coordinate_(&lagrange) {
    volume_ = this->BuildQuadraturePoints();
  }
  Pyramid(const Pyramid &) = default;
  Pyramid &operator=(const Pyramid &) = default;
  Pyramid(Pyramid &&) noexcept = default;
  Pyramid &operator=(Pyramid &&) noexcept = default;
  virtual ~Pyramid() noexcept = default;

  const Coordinate &coordinate() const final {
    return *coordinate_;
  }

  Scalar volume() const final {
    return volume_;
  }
};

template <class Gx, class Gy, int Qz>
typename Pyramid<Gx, Gy, Qz>::Points const
Pyramid<Gx, Gy, Qz>::local_coords_
    = Pyramid<Gx, Gy, Qz>::BuildLocalCoords();

template <class Gx, class Gy, int Qz>
typename Pyramid<Gx, Gy, Qz>::Weights const
Pyramid<Gx, Gy, Qz>::local_weights_
    = Pyramid<Gx, Gy, Qz>::BuildLocalWeights();

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_PYRAMID_HPP_
