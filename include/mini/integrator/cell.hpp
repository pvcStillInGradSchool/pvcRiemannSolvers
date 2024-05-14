//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_INTEGRATOR_CELL_HPP_
#define MINI_INTEGRATOR_CELL_HPP_

#include <concepts>

#include "mini/coordinate/cell.hpp"
#include "mini/integrator/element.hpp"
#include "mini/integrator/function.hpp"

namespace mini {
namespace integrator {

/**
 * @brief Abstract numerical integrators on volume elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Cell : public Element<Scalar, 3, 3> {
 public:
  using Coordinate = coordinate::Cell<Scalar>;
  using Real = typename Coordinate::Real;
  using Local = typename Coordinate::Local;
  using Global = typename Coordinate::Global;
  using Jacobian = typename Coordinate::Jacobian;

  virtual ~Cell() noexcept = default;
  virtual Real volume() const = 0;

  /**
   * @brief Get a reference to the coordinate::Cell object it uses for coordinate mapping.
   * 
   * @return const Coordinate &  Reference to the coordinate::Cell object it uses for coordinate mapping.
   */
  virtual const Coordinate &coordinate() const = 0;

  virtual std::unique_ptr<Cell> Clone(Coordinate const &) const = 0;
};

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_CELL_HPP_
