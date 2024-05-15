//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_LIMITER_RECONSTRUCT_HPP_
#define MINI_LIMITER_RECONSTRUCT_HPP_

#include <cassert>
#include <cmath>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "mini/basis/taylor.hpp"

namespace mini {
namespace limiter {

template <class Part, class Limiter>
void Reconstruct(Part *part_ptr, Limiter &&limiter) {
  if (Part::kDegrees == 0) {
    return;
  }
  using Cell = typename Part::Cell;
  using ProjectionWrapper
      = typename std::remove_reference_t<Limiter>::ProjectionWrapper;

  auto act = [&limiter](std::vector<Cell *> const &cell_ptrs) {
    auto troubled_cells = std::vector<Cell *>();
    for (Cell *cell_ptr : cell_ptrs) {
      if (limiter.IsNotSmooth(*cell_ptr)) {
        troubled_cells.push_back(cell_ptr);
      }
    }
    auto new_projections = std::vector<ProjectionWrapper>();
    for (Cell *cell_ptr : troubled_cells) {
      new_projections.emplace_back(limiter(*cell_ptr));
    }
    int i = 0;
    for (Cell *cell_ptr : troubled_cells) {
      cell_ptr->polynomial().SetCoeff(new_projections[i++].coeff());
    }
    assert(i == troubled_cells.size());
  };

  // run the limiter on inner cells that need no ghost cells
  part_ptr->ShareGhostCellCoeffs();
  act(part_ptr->GetInnerCellPointers());
  // run the limiter on inter cells that need ghost cells
  part_ptr->UpdateGhostCellCoeffs();
  act(part_ptr->GetInterCellPointers());
}

}  // namespace limiter
}  // namespace mini

#endif  // MINI_LIMITER_RECONSTRUCT_HPP_
