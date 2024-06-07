// Copyright 2024 PEI Weicheng
#ifndef TEST_MESH_SPATIAL_HPP_
#define TEST_MESH_SPATIAL_HPP_

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

void ResetRiemann() {
  using Jacobian = typename Riemann::Jacobian;
  Riemann::SetConvectionCoefficient(
    Jacobian{ {3., 0.}, {0., 4.} },
    Jacobian{ {5., 0.}, {0., 6.} },
    Jacobian{ {7., 0.}, {0., 8.} }
  );
  Riemann::Diffusion::SetProperty(1.0);
  Riemann::SetBetaValues(2.0, 1.0 / 12);
}

}  // namespace spatial
}  // namespace test

#endif  // TEST_MESH_SPATIAL_HPP_
