//  Copyright 2023 PEI Weicheng
#ifndef MINI_GAUSS_ELEMENT_HPP_
#define MINI_GAUSS_ELEMENT_HPP_

#include "mini/coordinate/element.hpp"
#include "mini/constant/index.hpp"

namespace mini {

/**
 * @brief Numerical integrators on regular elements.
 * 
 */
namespace integrator {

using namespace mini::constant::index;

/**
 * @brief Abstract numerical integrators on elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam kPhysDim  Dimension of the underlying physical space.
 * @tparam kCellDim  Dimension of the element as a manifold.
 */
template <std::floating_point Scalar, int kPhysDim, int kCellDim>
class Element {
 public:
  using Coordinate = coordinate::Element<Scalar, kPhysDim, kCellDim>;
  using Real = typename Coordinate::Real;
  using Local = typename Coordinate::Local;
  using Global = typename Coordinate::Global;
  using Jacobian = typename Coordinate::Jacobian;

  static constexpr int CellDim() { return kCellDim; }
  static constexpr int PhysDim() { return kPhysDim; }

  virtual ~Element() noexcept = default;

  /**
   * @brief Get the number of quadrature points on this element.
   * 
   * @return int  Number of quadrature points on this element.
   */
  virtual int CountPoints() const = 0;

  /**
   * @brief Get the Local of the i-th quadrature point.
   * 
   * @param i  0-based index of the i-th quadrature point.
   * @return const Local &  Local of the i-th quadrature point.
   */
  virtual const Local &GetLocal(int i) const = 0;

  /**
   * @brief Get the Global of the i-th quadrature point.
   * 
   * @param i  0-based index of the i-th quadrature point.
   * @return const Global &  Global of the i-th quadrature point.
   */
  virtual const Global &GetGlobal(int i) const = 0;

  /**
   * @brief Get the local (without Jacobian) weight of the i-th quadrature point.
   * 
   * @param i  0-based index of the i-th quadrature point.
   * @return const Real &  Local weight of the i-th quadrature point.
   */
  virtual const Real &GetLocalWeight(int i) const = 0;

  /**
   * @brief Get the global (with Jacobian) weight of the i-th quadrature point.
   * 
   * @param i  0-based index of the i-th quadrature point.
   * @return const Real &  Global weight of the i-th quadrature point.
   */
  virtual const Real &GetGlobalWeight(int i) const = 0;

  /**
   * @brief Get a reference to the coordinate::Element object it uses for coordinate mapping.
   * 
   * @return const Coordinate &  Reference to the coordinate::Element object it uses for coordinate mapping.
   */
  virtual const Coordinate &coordinate() const = 0;

  const Global &center() const {
    return coordinate().center();
  }

 protected:
  virtual Global &GetGlobal(int i) = 0;
  virtual Real &GetGlobalWeight(int i) = 0;
  Real BuildQuadraturePoints() {
    Real sum = 0.0;
    for (int i = 0, n = CountPoints(); i < n; ++i) {
      auto &local_i = GetLocal(i);
      GetGlobal(i) = coordinate().LocalToGlobal(local_i);
      auto mat_j = coordinate().LocalToJacobian(local_i);
      auto det_j = CellDim() < PhysDim()
          ? std::sqrt((mat_j * mat_j.transpose()).determinant())
          : std::abs(mat_j.determinant());
      GetGlobalWeight(i) = GetLocalWeight(i) * det_j;
      sum += GetGlobalWeight(i);
    }
    return sum;
  }
};

}  // namespace integrator
}  // namespace mini

#endif  // MINI_GAUSS_ELEMENT_HPP_
