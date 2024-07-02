//  Copyright 2024 PEI Weicheng
#ifndef MINI_LIMITER_AVERAGE_HPP_
#define MINI_LIMITER_AVERAGE_HPP_

#include "mini/polynomial/concept.hpp"

namespace mini {
namespace limiter {
namespace average {

template <class Part>
    requires(mini::polynomial::General<typename Part::Cell::Polynomial>)
void Reconstruct(Part *part_ptr) {
  if (!Part::kDegrees) {
    return;
  }
  using Cell = typename Part::Cell;
  using Polynomial = typename Cell::Polynomial;
  using Value = typename Polynomial::Value;

  for (Cell *cell_ptr : part_ptr->GetLocalCellPointers()) {
    Polynomial *polynomial_ptr = &cell_ptr->polynomial();
    Value average = polynomial_ptr->average();
    for (int q = 0; q < Cell::N; ++q) {
      polynomial_ptr->SetValue(q, average);
    }
  }
}

}  // namespace average
}  // namespace limiter
}  // namespace mini

#endif  // MINI_LIMITER_AVERAGE_HPP_
