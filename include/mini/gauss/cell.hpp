//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_GAUSS_CELL_HPP_
#define MINI_GAUSS_CELL_HPP_

#include <concepts>

#include "mini/geometry/cell.hpp"
#include "mini/gauss/element.hpp"
#include "mini/gauss/function.hpp"

namespace mini {
namespace gauss {

/**
 * @brief Abstract numerical integrators on volume elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Cell : public Element<Scalar, 3, 3> {
 public:
  using Lagrange = geometry::Cell<Scalar>;
  using Real = typename Lagrange::Real;
  using Local = typename Lagrange::Local;
  using Global = typename Lagrange::Global;
  using Jacobian = typename Lagrange::Jacobian;

  virtual ~Cell() noexcept = default;
  virtual Real volume() const = 0;

  /**
   * @brief Get a reference to the geometry::Cell object it uses for coordinate mapping.
   * 
   * @return const Lagrange &  Reference to the geometry::Cell object it uses for coordinate mapping.
   */
  virtual const Lagrange &coordinate() const = 0;
};

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_CELL_HPP_
