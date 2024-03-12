//  Copyright 2024 PEI Weicheng
#ifndef DEMO_LAMINAR_SHOCKLESS_HPP_
#define DEMO_LAMINAR_SHOCKLESS_HPP_

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "mpi.h"
#include "pcgnslib.h"

#include "mini/riemann/concept.hpp"
#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/ausm.hpp"
#include "mini/riemann/euler/hllc.hpp"
#include "mini/riemann/euler/exact.hpp"
#include "mini/riemann/rotated/euler.hpp"
#include "mini/riemann/diffusive/navier_stokes.hpp"
#include "mini/riemann/diffusive/direct_dg.hpp"
#include "mini/polynomial/hexahedron.hpp"
#include "mini/mesh/part.hpp"
#include "mini/temporal/rk.hpp"
#include "mini/spatial/dg/lobatto.hpp"
#include "mini/spatial/fr/lobatto.hpp"

#define FR

using Scalar = double;

/* Define the Navier--Stokes equation. */
constexpr int kDimensions = 3;

extern template class mini::riemann::euler::IdealGas<Scalar, 1.4>;
using Gas = mini::riemann::euler::IdealGas<Scalar, 1.4>;

extern template class mini::riemann::euler::Primitives<Scalar, kDimensions>;
using Primitive = mini::riemann::euler::Primitives<Scalar, kDimensions>;

extern template class mini::riemann::euler::Conservatives<Scalar, kDimensions>;
using Conservative = mini::riemann::euler::Conservatives<Scalar, kDimensions>;

extern template class mini::riemann::euler::Hllc<Gas, kDimensions>;
using Unrotated = mini::riemann::euler::Hllc<Gas, kDimensions>;

extern template class mini::riemann::rotated::Euler<Unrotated>;
using Convection = mini::riemann::rotated::Euler<Unrotated>;

extern template class mini::riemann::diffusive::NavierStokes<Gas>;
using NavierStokes =  mini::riemann::diffusive::NavierStokes<Gas>;

extern template class mini::riemann::diffusive::DirectDG<NavierStokes>;
using Diffusion = mini::riemann::diffusive::DirectDG<NavierStokes>;

extern template class mini::riemann::ConvectionDiffusion<Convection, Diffusion>;
using Riemann = mini::riemann::ConvectionDiffusion<Convection, Diffusion>;

/* Define spatial discretization. */
constexpr int kDegrees = 2;
constexpr int kComponents = Riemann::kComponents;
using Gx = mini::gauss::Lobatto<Scalar, kDegrees + 1>;
#ifdef DG
using Projection = mini::polynomial::Hexahedron<Gx, Gx, Gx, kComponents,
    false>;
#endif
#ifdef FR
using Projection = mini::polynomial::Hexahedron<Gx, Gx, Gx, kComponents,
    true>;
#endif

using Part = mini::mesh::part::Part<cgsize_t, Riemann, Projection>;
using Cell = typename Part::Cell;
using Face = typename Part::Face;
using Global = typename Cell::Global;
using Value = typename Cell::Value;
using Coeff = typename Cell::Coeff;

#ifdef DG
using Spatial = mini::spatial::dg::Lobatto<Part>;
#endif
#ifdef FR
using Spatial = mini::spatial::fr::Lobatto<Part>;
#endif

/* Choose the time-stepping scheme. */
constexpr int kOrders = std::min(3, kDegrees + 1);
using Temporal = mini::temporal::RungeKutta<kOrders, Scalar>;

/* Define the types of IC and BCs. */
using IC = Value(*)(const Global &);
using BC = void(*)(const std::string &, Spatial *);

int Main(int argc, char* argv[], IC ic, BC bc);

#endif  // DEMO_LAMINAR_SHOCKLESS_HPP_
