//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_INTEGRATOR_HEXAHEDRON_HPP_
#define MINI_INTEGRATOR_HEXAHEDRON_HPP_

#include <concepts>

#include <cassert>
#include <cmath>
#include <cstring>

#include <algorithm>
#include <type_traits>

#include "mini/integrator/line.hpp"
#include "mini/integrator/cell.hpp"
#include "mini/coordinate/hexahedron.hpp"

namespace mini {
namespace integrator {
/**
 * @brief Numerical integrators on hexahedral elements.
 * 
 * @tparam Gx  The quadrature rule in the \f$\xi\f$ direction.
 * @tparam Gy  The quadrature rule in the \f$\eta\f$ direction.
 * @tparam Gz  The quadrature rule in the \f$\zeta\f$ direction.
 */
template <class Gx, class Gy, class Gz>
class Hexahedron : public Cell<typename Gx::Scalar> {
 public:
  static_assert(std::is_same_v<typename Gx::Scalar, typename Gy::Scalar>);
  static_assert(std::is_same_v<typename Gx::Scalar, typename Gz::Scalar>);
  using IntegratorX = Gx;
  using IntegratorY = Gy;
  using IntegratorZ = Gz;
  using Scalar = typename IntegratorX::Scalar;
  static_assert(std::is_same_v<Scalar, typename Gy::Scalar>);
  static_assert(std::is_same_v<Scalar, typename Gz::Scalar>);
  using Coordinate = coordinate::Hexahedron<Scalar>;
  using Local = typename Coordinate::Local;
  using Global = typename Coordinate::Global;
  using Jacobian = typename Coordinate::Jacobian;
  using Base = Cell<Scalar>;

 private:
  static constexpr int Qx = IntegratorX::Q;
  static constexpr int Qy = IntegratorY::Q;
  static constexpr int Qz = IntegratorZ::Q;

 public:
  static constexpr int Q = Qx * Qy * Qz;

 private:
  static const std::array<Local, Q> local_coords_;
  static const std::array<Scalar, Q> local_weights_;
  std::array<Global, Q> global_coords_;
  std::array<Scalar, Q> global_weights_;
  Coordinate const *coordinate_;
  Scalar volume_;


 public:
  int CountPoints() const final {
    return Q;
  }

 private:
  static constexpr auto BuildLocalCoords() {
    std::array<Local, Qx * Qy * Qz> points;
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
    std::array<Scalar, Qx * Qy * Qz> weights;
    int n = 0;
    for (int i = 0; i < Qx; ++i) {
      for (int j = 0; j < Qy; ++j) {
        for (int k = 0; k < Qz; ++k) {
          weights[n++] = IntegratorX::weights[i] * IntegratorY::weights[j]
              * IntegratorZ::weights[k];
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
    return std::make_unique<Hexahedron>(
        dynamic_cast<Coordinate const &>(coordinate));
  }

  explicit Hexahedron(Coordinate const &lagrange)
      : coordinate_(&lagrange) {
    volume_ = this->BuildQuadraturePoints();
  }
  Hexahedron(const Hexahedron &) = default;
  Hexahedron &operator=(const Hexahedron &) = default;
  Hexahedron(Hexahedron &&) noexcept = default;
  Hexahedron &operator=(Hexahedron &&) noexcept = default;
  virtual ~Hexahedron() noexcept = default;

  const Coordinate &coordinate() const final {
    return *coordinate_;
  }

  Scalar volume() const final {
    return volume_;
  }
};

template <class Gx, class Gy, class Gz>
std::array<typename Hexahedron<Gx, Gy, Gz>::Local,
    Hexahedron<Gx, Gy, Gz>::Q> const
Hexahedron<Gx, Gy, Gz>::local_coords_
    = Hexahedron<Gx, Gy, Gz>::BuildLocalCoords();

template <class Gx, class Gy, class Gz>
std::array<typename Hexahedron<Gx, Gy, Gz>::Scalar,
    Hexahedron<Gx, Gy, Gz>::Q> const
Hexahedron<Gx, Gy, Gz>::local_weights_
    = Hexahedron<Gx, Gy, Gz>::BuildLocalWeights();

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_HEXAHEDRON_HPP_
