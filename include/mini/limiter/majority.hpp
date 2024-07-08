//  Copyright 2024 PEI Weicheng
#ifndef MINI_LIMITER_MAJORITY_HPP_
#define MINI_LIMITER_MAJORITY_HPP_

#include <cassert>
#include <unordered_map>

#include "mini/polynomial/concept.hpp"

namespace mini {
namespace limiter {
namespace majority {

template <class Part>
    requires(mini::polynomial::Nodal<typename Part::Cell::Polynomial>)
void Reconstruct(Part *part_ptr) {
  if (Part::kDegrees < 2) {
    return;
  }
  using Cell = typename Part::Cell;
  using Polynomial = typename Cell::Polynomial;
  using Value = typename Polynomial::Value;

  for (Cell *cell_ptr : part_ptr->GetLocalCellPointers()) {
    Polynomial *polynomial_ptr = &cell_ptr->polynomial();
    Value value;
    for (int k = 0; k < Cell::K; ++k) {
      auto val_to_cnt = std::unordered_map<Scalar, int>();
      for (int q = 0; q < Cell::N; ++q) {
        Scalar val = polynomial_ptr->GetValue(q)[k];
        if (++val_to_cnt[val] * 2 > Cell::N) {
          value[k] = val;
          break;
        }
      }
      assert(val_to_cnt[value[k]] * 2 > Cell::N);
    }
    for (int q = 0; q < Cell::N; ++q) {
      polynomial_ptr->SetValue(q, value);
    }
  }
}

}  // namespace majority
}  // namespace limiter
}  // namespace mini

#endif  // MINI_LIMITER_MAJORITY_HPP_
