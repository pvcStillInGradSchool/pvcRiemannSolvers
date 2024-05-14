//  Copyright 2023 PEI Weicheng
#ifndef MINI_INTEGRATOR_WEDGE_HPP_
#define MINI_INTEGRATOR_WEDGE_HPP_

#include <concepts>

#include <cassert>
#include <cmath>
#include <cstring>

#include <algorithm>
#include <type_traits>

#include "mini/integrator/line.hpp"
#include "mini/integrator/cell.hpp"
#include "mini/integrator/triangle.hpp"
#include "mini/coordinate/wedge.hpp"

namespace mini {
namespace integrator {
/**
 * @brief Numerical integrators on hexahedral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam Qt  Number of qudrature points in each layer of triangle.
 * @tparam Qz  Number of qudrature points in the \f$\zeta\f$ direction.
 * @tparam kRule  The type of Integratorian quadrature rule.
 */
template <std::floating_point Scalar, int Qt, int Qz,
    Rule kRule = Rule::kLegendre>
class Wedge : public Cell<Scalar> {
  static constexpr int kPoints = Qt * Qz;

 public:
  using Base = Cell<Scalar>;
  using IntegratorT = Triangle<Scalar, 2, Qt>;
  using IntegratorZ = std::conditional_t< kRule == Rule::kLegendre,
      Legendre<Scalar, Qz>, Lobatto<Scalar, Qz> >;
  using Coordinate = coordinate::Wedge<Scalar>;
  using Real = typename Coordinate::Real;
  using Local = typename Coordinate::Local;
  using Global = typename Coordinate::Global;
  using Jacobian = typename Coordinate::Jacobian;

 private:
  static const std::array<Local, Qt * Qz> local_coords_;
  static const std::array<Scalar, Qt * Qz> local_weights_;
  std::array<Global, kPoints> global_coords_;
  std::array<Scalar, kPoints> global_weights_;
  Coordinate const *coordinate_;
  Scalar volume_;

 public:
  int CountPoints() const final {
    return kPoints;
  }

 private:
  static constexpr auto BuildLocalCoords() {
    std::array<Local, kPoints> points;
    auto triangle_points = IntegratorT::BuildLocalCoords();
    int n = 0;
    for (int i = 0; i < Qt; ++i) {
      for (int k = 0; k < Qz; ++k) {
        points[n][X] = triangle_points[i][X];
        points[n][Y] = triangle_points[i][Y];
        points[n][Z] = IntegratorZ::points[k];
        n++;
      }
    }
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, Qt * Qz> weights;
    auto triangle_weights = IntegratorT::BuildLocalWeights();
    int n = 0;
    for (int i = 0; i < Qt; ++i) {
      for (int k = 0; k < Qz; ++k) {
        weights[n++] = triangle_weights[i] * IntegratorZ::weights[k];
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
    return std::make_unique<Wedge>(
        dynamic_cast<Coordinate const &>(coordinate));
  }

  explicit Wedge(Coordinate const &lagrange)
      : coordinate_(&lagrange) {
    volume_ = this->BuildQuadraturePoints();
  }
  Wedge(const Wedge &) = default;
  Wedge &operator=(const Wedge &) = default;
  Wedge(Wedge &&) noexcept = default;
  Wedge &operator=(Wedge &&) noexcept = default;
  virtual ~Wedge() noexcept = default;

  const Coordinate &coordinate() const final {
    return *coordinate_;
  }

  Scalar volume() const final {
    return volume_;
  }
};

template <std::floating_point Scalar, int Qt, int Qz, Rule R>
std::array<typename Wedge<Scalar, Qt, Qz, R>::Local, Qt * Qz> const
Wedge<Scalar, Qt, Qz, R>::local_coords_
    = Wedge<Scalar, Qt, Qz, R>::BuildLocalCoords();

template <std::floating_point Scalar, int Qt, int Qz, Rule R>
std::array<Scalar, Qt * Qz> const
Wedge<Scalar, Qt, Qz, R>::local_weights_
    = Wedge<Scalar, Qt, Qz, R>::BuildLocalWeights();

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_WEDGE_HPP_
