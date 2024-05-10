// Copyright 2023 PEI Weicheng
#ifndef MINI_SPATIAL_LIMITED_HPP_
#define MINI_SPATIAL_LIMITED_HPP_

#include "mini/spatial/fem.hpp"

namespace mini {
namespace spatial {

template <typename Part, typename Limiter>
class Limited : public FiniteElement<Part> {
  using Base = FiniteElement<Part>;

 public:
  using Riemann = typename Base::Riemann;
  using Scalar = typename Base::Scalar;
  using Face = typename Base::Face;
  using Cell = typename Base::Cell;
  using Global = typename Base::Global;
  using Projection = typename Base::Projection;
  using Coeff = typename Base::Coeff;
  using Value = typename Base::Value;
  using Temporal = typename Base::Temporal;
  using Column = typename Base::Column;

 protected:
  Limiter limiter_;

 public:
  Limited(Part *part_ptr,
          const Limiter &limiter)
      : Base(part_ptr), limiter_(limiter) {
  }
  Limited(const Limited &) = default;
  Limited &operator=(const Limited &) = default;
  Limited(Limited &&) noexcept = default;
  Limited &operator=(Limited &&) noexcept = default;
  ~Limited() noexcept = default;

 public:  // implement pure virtual methods declared in Temporal
  void SetSolutionColumn(Column const &column) override {
    this->Base::SetSolutionColumn(column);
    mini::limiter::Reconstruct(this->part_ptr_, limiter_);
  }
};

}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_LIMITED_HPP_
