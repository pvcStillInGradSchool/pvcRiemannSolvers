//  Copyright 2023 PEI Weicheng
#ifndef MINI_INTEGRATOR_WEDGE_HPP_
#define MINI_INTEGRATOR_WEDGE_HPP_

#include <concepts>

#include <cassert>
#include <cmath>
#include <cstring>

#include <algorithm>
#include <type_traits>

#include "mini/integrator/cell.hpp"
#include "mini/integrator/triangle.hpp"
#include "mini/coordinate/wedge.hpp"

namespace mini {
namespace integrator {
/**
 * @brief Numerical integrators on hexahedral elements.
 * 
 * @tparam Qt  Number of qudrature points in each layer of triangle.
 * @tparam Gx  The quadrature rule in the \f$\zeta\f$ direction.
 */
template <int Qt, class Gx>
class Wedge : public Cell<typename Gx::Scalar> {
 public:
  using Scalar = typename Gx::Scalar;
  using Base = Cell<Scalar>;
  using IntegratorT = Triangle<Scalar, 2, Qt>;
  using IntegratorZ = Gx;
  using Coordinate = coordinate::Wedge<Scalar>;
  using Real = typename Coordinate::Real;
  using Local = typename Coordinate::Local;
  using Global = typename Coordinate::Global;
  static_assert(std::is_same_v<Local, Global>);
  using Jacobian = typename Coordinate::Jacobian;

  static constexpr int Q = Qt * Gx::Q;

 private:
  using Points = std::array<Local, Q>;
  static const Points local_coords_;

  using Weights = std::array<Scalar, Q>;
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
    auto triangle_points = IntegratorT::BuildLocalCoords();
    int n = 0;
    for (int i = 0; i < Qt; ++i) {
      for (int k = 0; k < IntegratorZ::Q; ++k) {
        points[n][X] = triangle_points[i][X];
        points[n][Y] = triangle_points[i][Y];
        points[n][Z] = IntegratorZ::points[k];
        n++;
      }
    }
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    Weights weights;
    auto triangle_weights = IntegratorT::BuildLocalWeights();
    int n = 0;
    for (int i = 0; i < Qt; ++i) {
      for (int k = 0; k < IntegratorZ::Q; ++k) {
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

template <int Qt, class Gz>
typename Wedge<Qt, Gz>::Points const
Wedge<Qt, Gz>::local_coords_
    = Wedge<Qt, Gz>::BuildLocalCoords();

template <int Qt, class Gz>
typename Wedge<Qt, Gz>::Weights const
Wedge<Qt, Gz>::local_weights_
    = Wedge<Qt, Gz>::BuildLocalWeights();

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_WEDGE_HPP_
