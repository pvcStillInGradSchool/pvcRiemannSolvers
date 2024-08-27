//  Copyright 2024 PEI Weicheng
#ifndef DEMO_LAMINAR_SHOCKED_HPP_
#define DEMO_LAMINAR_SHOCKED_HPP_

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "mpi.h"
#include "pcgnslib.h"

#include "mini/riemann/diffusive/navier_stokes.hpp"
#include "mini/riemann/diffusive/direct.hpp"
#include "mini/polynomial/hexahedron.hpp"
#include "mini/mesh/part.hpp"
#include "mini/temporal/rk.hpp"
#include "mini/spatial/dg/lobatto.hpp"
#include "mini/spatial/fr/lobatto.hpp"
#include "mini/spatial/viscosity.hpp"
#include "mini/spatial/with_viscosity.hpp"

#define FR // exactly one of (DGFEM, DGSEM, FR) must be defined
#define VISCOSITY  // one of (LIMITER, VISCOSITY) must be defined

using Scalar = double;

/* Define the Navier--Stokes equation. */
constexpr int kDimensions = 3;
#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/ausm.hpp"
#include "mini/riemann/euler/hllc.hpp"
#include "mini/riemann/euler/exact.hpp"
#include "mini/riemann/rotated/euler.hpp"
#include "mini/riemann/diffusive/navier_stokes.hpp"
// extern template class mini::riemann::euler::IdealGas<Scalar, 1.4>;
using Gas = mini::riemann::euler::IdealGas<Scalar, 1.4>;

// extern template class mini::riemann::euler::Primitives<Scalar, kDimensions>;
using Primitive = mini::riemann::euler::Primitives<Scalar, kDimensions>;

// extern template class mini::riemann::euler::Conservatives<Scalar, kDimensions>;
using Conservative = mini::riemann::euler::Conservatives<Scalar, kDimensions>;

// extern template class mini::riemann::euler::HartenLaxLeerContact<Gas, kDimensions>;
using Unrotated = mini::riemann::euler::HartenLaxLeerContact<Gas, kDimensions>;

// extern template class mini::riemann::rotated::Euler<Unrotated>;
using Convection = mini::riemann::rotated::Euler<Unrotated>;

// extern template class mini::riemann::diffusive::NavierStokes<Gas>;
using NavierStokes =  mini::riemann::diffusive::NavierStokes<Gas>;

// extern template class mini::riemann::diffusive::Direct<NavierStokes>;
using Diffusion = mini::riemann::diffusive::Direct<NavierStokes>;

// extern template class mini::riemann::ConvectionDiffusion<Convection, Diffusion>;
using Riemann = mini::riemann::ConvectionDiffusion<Convection, Diffusion>;

/* Define polynomial approximation. */
constexpr int kDegrees = 2;
constexpr int kComponents = Riemann::kComponents;
#ifdef DGFEM
#include "mini/integrator/legendre.hpp"
using Gx = mini::integrator::Legendre<Scalar, kDegrees + 1>;
#include "mini/polynomial/projection.hpp"
using Polynomial = mini::polynomial::Projection<Scalar, kDimensions, kDegrees, kComponents>;

#else  // common for DGSEM and FR
#include "mini/integrator/lobatto.hpp"
using Gx = mini::integrator::Lobatto<Scalar, kDegrees + 1>;

#include "mini/polynomial/hexahedron.hpp"
using Interpolation = mini::polynomial::Hexahedron<Gx, Gx, Gx, kComponents, false>;

#ifdef LIMITER
#include "mini/polynomial/extrapolation.hpp"
using Polynomial = mini::polynomial::Extrapolation<Interpolation>;

#else  // VISCOSITY
using Polynomial = Interpolation;
#endif  // LIMITER

#endif  // DGSEM

using Part = mini::mesh::part::Part<cgsize_t, Polynomial>;
using Cell = typename Part::Cell;
using Face = typename Part::Face;
using Global = typename Cell::Global;
using Value = typename Cell::Value;
using Coeff = typename Cell::Coeff;

static void InstallIntegratorPrototypes(Part *part_ptr);

/* Chose the spatial scheme and the method for shock capturing. */
#ifdef VISCOSITY
#include "mini/riemann/concept.hpp"
#include "mini/spatial/viscosity.hpp"
#include "mini/spatial/with_viscosity.hpp"
// extern template class mini::spatial::EnergyBasedViscosity<Part, Riemann>;
using RiemannWithViscosity = mini::spatial::EnergyBasedViscosity<Part, Riemann>;
static_assert(mini::riemann::ConvectiveDiffusive<RiemannWithViscosity>);

#if defined(FR)
#include "mini/spatial/fr/lobatto.hpp"
using General = mini::spatial::fr::Lobatto<Part, RiemannWithViscosity>;
#endif

using Spatial = mini::spatial::WithViscosity<General>;

#endif  // VISCOSITY

/* Choose the time-stepping scheme. */
constexpr int kOrders = std::min(3, kDegrees + 1);
using Temporal = mini::temporal::RungeKutta<kOrders, Scalar>;

/* Define the types of IC and BCs. */
using IC = Value(*)(const Global &);
using BC = void(*)(const std::string &, Spatial *);

int Main(int argc, char* argv[], IC ic, BC bc);

#endif  // DEMO_LAMINAR_SHOCKED_HPP_
