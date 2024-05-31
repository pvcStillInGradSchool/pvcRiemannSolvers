// Copyright 2023 PEI Weicheng
#ifndef MINI_SPATIAL_WITH_LIMITER_HPP_
#define MINI_SPATIAL_WITH_LIMITER_HPP_

#include <concepts>

#include "mini/spatial/fem.hpp"

namespace mini {
namespace spatial {

/**
 * @brief Augment a concrete FiniteElement with a concrete Limiter.
 * 
 */
template <typename ConcreteFiniteElement, typename Limiter>
class WithLimiter : public ConcreteFiniteElement {
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

 protected:
  Limiter *limiter_ptr_;

 public:
  template <class... Args>
  WithLimiter(Limiter *limiter_ptr, Args&&... args)
      : Base(std::forward<Args>(args)...), limiter_ptr_(limiter_ptr) {
  }
  WithLimiter(const WithLimiter &) = default;
  WithLimiter &operator=(const WithLimiter &) = default;
  WithLimiter(WithLimiter &&) noexcept = default;
  WithLimiter &operator=(WithLimiter &&) noexcept = default;
  ~WithLimiter() noexcept = default;

  Limiter *limiter_ptr() const {
    return limiter_ptr_;
  }

 public:  // implement pure virtual methods declared in Temporal
  void SetSolutionColumn(Column const &column) override {
    this->Base::SetSolutionColumn(column);
    mini::limiter::Reconstruct(this->part_ptr_, limiter_ptr_);
  }
};

}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_WITH_LIMITER_HPP_
