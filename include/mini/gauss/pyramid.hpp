//  Copyright 2023 PEI Weicheng
#ifndef MINI_GAUSS_PYRAMID_HPP_
#define MINI_GAUSS_PYRAMID_HPP_

#include <concepts>

#include <cassert>
#include <cmath>

#include <algorithm>
#include <type_traits>

#include "mini/gauss/line.hpp"
#include "mini/gauss/jacobi.hpp"
#include "mini/gauss/cell.hpp"
#include "mini/geometry/pyramid.hpp"

namespace mini {
namespace gauss {
/**
 * @brief Numerical integrators on pyramidal elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam Qx  Number of qudrature points in the \f$\xi\f$ direction.
 * @tparam Qy  Number of qudrature points in the \f$\eta\f$ direction.
 * @tparam Qz  Number of qudrature points in the \f$\zeta\f$ direction.
 * @tparam kRule  The type of Gaussian quadrature rule.
 */
template <std::floating_point Scalar, int Qx, int Qy, int Qz,
    Rule kRule = Rule::kLegendre>
class Pyramid : public Cell<Scalar> {
 public:
  using GaussX = std::conditional_t< kRule == Rule::kLegendre,
      Legendre<Scalar, Qx>, Lobatto<Scalar, Qx> >;
  using GaussY = std::conditional_t< kRule == Rule::kLegendre,
      Legendre<Scalar, Qy>, Lobatto<Scalar, Qy> >;
  using GaussZ = Jacobi<Scalar, Qz, 2, 0>;
  using Coordinate = geometry::Pyramid<Scalar>;
  using Real = typename Coordinate::Real;
  using Local = typename Coordinate::Local;
  using Global = typename Coordinate::Global;
  using Jacobian = typename Coordinate::Jacobian;

 private:
  static const std::array<Local, Qx * Qy * Qz> local_coords_;
  static const std::array<Scalar, Qx * Qy * Qz> local_weights_;
  std::array<Global, Qx * Qy * Qz> global_coords_;
  std::array<Scalar, Qx * Qy * Qz> global_weights_;
  Coordinate const *lagrange_;
  Scalar volume_;

 public:
  int CountPoints() const final {
    return Qx * Qy * Qz;
  }

 private:
  static constexpr auto BuildLocalCoords() {
    std::array<Local, Qx * Qy * Qz> points;
    int n = 0;
    for (int i = 0; i < Qx; ++i) {
      for (int j = 0; j < Qy; ++j) {
        for (int k = 0; k < Qz; ++k) {
          points[n][X] = GaussX::points[i];
          points[n][Y] = GaussY::points[j];
          points[n][Z] = GaussZ::points[k];
          n++;
        }
      }
    }
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, Qx * Qy * Qz> weights;
    int n = 0;
    for (int i = 0; i < Qx; ++i) {
      for (int j = 0; j < Qy; ++j) {
        for (int k = 0; k < Qz; ++k) {
          weights[n++] = GaussX::weights[i] * GaussY::weights[j]
              // Jacobi::weights have taken account of \f$ (1 - \xi)^2 \f$.
              * GaussZ::weights[k] * std::pow(1 - GaussZ::points[k], -2);
        }
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
  explicit Pyramid(Coordinate const &lagrange)
      : lagrange_(&lagrange) {
    volume_ = this->BuildQuadraturePoints();
  }
  Pyramid(const Pyramid &) = default;
  Pyramid &operator=(const Pyramid &) = default;
  Pyramid(Pyramid &&) noexcept = default;
  Pyramid &operator=(Pyramid &&) noexcept = default;
  virtual ~Pyramid() noexcept = default;

  const Coordinate &coordinate() const final {
    return *lagrange_;
  }

  Scalar volume() const final {
    return volume_;
  }
};

template <std::floating_point Scalar, int Qx, int Qy, int Qz, Rule R>
std::array<typename Pyramid<Scalar, Qx, Qy, Qz, R>::Local, Qx * Qy * Qz> const
Pyramid<Scalar, Qx, Qy, Qz, R>::local_coords_
    = Pyramid<Scalar, Qx, Qy, Qz, R>::BuildLocalCoords();

template <std::floating_point Scalar, int Qx, int Qy, int Qz, Rule R>
std::array<Scalar, Qx * Qy * Qz> const
Pyramid<Scalar, Qx, Qy, Qz, R>::local_weights_
    = Pyramid<Scalar, Qx, Qy, Qz, R>::BuildLocalWeights();

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_PYRAMID_HPP_
