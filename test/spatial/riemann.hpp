// Copyright 2024 PEI Weicheng
#ifndef TEST_SPATIAL_RIEMANN_HPP_
#define TEST_SPATIAL_RIEMANN_HPP_

#include <cassert>
#include <cmath>

#include <algorithm>

#include "mini/riemann/concept.hpp"
#include "mini/riemann/rotated/multiple.hpp"
#include "mini/riemann/diffusive/linear.hpp"
#include "mini/riemann/diffusive/direct_dg.hpp"

#include "test/mesh/part.hpp"

namespace test {
namespace spatial {

using Convection = mini::
    riemann::rotated::Multiple<Scalar, kComponents, kDimensions>;
using Diffusion = mini::riemann::diffusive::DirectDG<
    mini::riemann::diffusive::Isotropic<Scalar, kComponents>
>;
using Riemann = mini::riemann::ConvectionDiffusion<Convection, Diffusion>;
static_assert(mini::riemann::ConvectiveDiffusive<Riemann>);

void ResetRiemann() {
  using Jacobian = typename Riemann::Jacobian;
  Riemann::Convection::SetJacobians(
    Jacobian{ {3., 0.}, {0., 4.} },
    Jacobian{ {5., 0.}, {0., 6.} },
    Jacobian{ {7., 0.}, {0., 8.} });
  assert(std::max(std::hypot(3., 5., 7.), std::hypot(4., 6., 8.))
      == Riemann::Convection::GetMaximumSpeed(Value::Random()));
  Riemann::Diffusion::SetProperty(1.0);
  Riemann::Diffusion::SetBetaValues(2.0, 1.0 / 12);
}

}  // namespace spatial
}  // namespace test

#endif  // TEST_SPATIAL_RIEMANN_HPP_
