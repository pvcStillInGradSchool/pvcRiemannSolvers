// Copyright 2024 PEI Weicheng
#ifndef MINI_SPATIAL_WITH_VISCOSITY_HPP_
#define MINI_SPATIAL_WITH_VISCOSITY_HPP_

#include <concepts>

#include "mini/spatial/fem.hpp"

namespace mini {
namespace spatial {

/**
 * @brief Augment a concrete FiniteElement with a concrete Viscosity.
 * 
 */
template <typename ConcreteFiniteElement>
class WithViscosity : public ConcreteFiniteElement {
 public:
  using Base = ConcreteFiniteElement;
  using Part = typename Base::Part;
  using Riemann = typename Base::Riemann;
  static_assert(std::derived_from<Base, FiniteElement<Part, Riemann>>);

  using Scalar = typename Base::Scalar;
  using Face = typename Base::Face;
  using Cell = typename Base::Cell;
  using Global = typename Base::Global;
  using Coeff = typename Base::Coeff;
  using Value = typename Base::Value;
  using Temporal = typename Base::Temporal;
  using Column = typename Base::Column;

 public:
  template <class... Args>
  WithViscosity(Args&&... args)
      : Base(std::forward<Args>(args)...) {
    Riemann::Viscosity::Initialize(this);
  }
  WithViscosity(const WithViscosity &) = default;
  WithViscosity &operator=(const WithViscosity &) = default;
  WithViscosity(WithViscosity &&) noexcept = default;
  WithViscosity &operator=(WithViscosity &&) noexcept = default;
  ~WithViscosity() noexcept = default;

 public:  // override virtual methods declared in ConcreteFiniteElement
  Column GetResidualColumn() const override {
    Riemann::Viscosity::UpdateProperties();
    // TODO(PVC): overlap communication with computation
    Riemann::Viscosity::ShareGhostCellProperties();
    Riemann::Viscosity::UpdateGhostCellProperties();
    return this->Base::GetResidualColumn();
  }
};

}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_WITH_VISCOSITY_HPP_
