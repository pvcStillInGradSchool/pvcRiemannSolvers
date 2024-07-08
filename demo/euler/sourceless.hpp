//  Copyright 2022 PEI Weicheng
#ifndef DEMO_EULER_SOURCELESS_HPP_
#define DEMO_EULER_SOURCELESS_HPP_

#include <algorithm>
#include <string>

#define FR // exactly one of (DGFEM, DGSEM, FR) must be defined
#define VISCOSITY  // one of (LIMITER, VISCOSITY) must be defined

using Scalar = double;

/* Define the Euler system. */
constexpr int kDimensions = 3;
#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/eigen.hpp"
#include "mini/riemann/euler/exact.hpp"
#include "mini/riemann/rotated/euler.hpp"
using Primitive = mini::riemann::euler::Primitives<Scalar, kDimensions>;
using Conservative = mini::riemann::euler::Conservatives<Scalar, kDimensions>;
using Gas = mini::riemann::euler::IdealGas<Scalar, 1.4>;
using Unrotated = mini::riemann::euler::Exact<Gas, kDimensions>;
using Riemann = mini::riemann::rotated::Euler<Unrotated>;

/* Define polynomial approximation. */
constexpr int kComponents = 5;
constexpr int kDegrees = 2;
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

#include "mini/mesh/part.hpp"
using Part = mini::mesh::part::Part<cgsize_t, Polynomial>;
using Cell = typename Part::Cell;
using Face = typename Part::Face;
using Global = typename Cell::Global;
using Value = typename Cell::Value;
using Coeff = typename Cell::Coeff;

static void InstallIntegratorPrototypes(Part *part_ptr);

/* Chose the spatial scheme and the method for shock capturing. */
#ifdef LIMITER

#ifdef DGFEM
#include "mini/spatial/dg/general.hpp"
using General = mini::spatial::dg::General<Part, Riemann>;

#elif defined(DGSEM)
#include "mini/spatial/dg/lobatto.hpp"
using General = mini::spatial::dg::Lobatto<Part, Riemann>;

#elif defined(FR)
#include "mini/spatial/fr/lobatto.hpp"
using General = mini::spatial::fr::Lobatto<Part, Riemann>;
#endif

#include "mini/limiter/weno.hpp"
#include "mini/limiter/reconstruct.hpp"
#include "mini/spatial/with_limiter.hpp"
using Limiter = mini::limiter::weno::Lazy<Cell>;
using Spatial = mini::spatial::WithLimiter<General, Limiter>;

#endif  // LIMITER

#ifdef VISCOSITY

#include "mini/riemann/concept.hpp"
#include "mini/riemann/diffusive/linear.hpp"
#include "mini/riemann/diffusive/direct.hpp"
#include "mini/spatial/viscosity.hpp"
#include "mini/spatial/with_viscosity.hpp"

using Diffusion = mini::riemann::diffusive::Direct<
    mini::riemann::diffusive::Isotropic<Scalar, kComponents>>;
using RiemannWithViscosity = mini::spatial::EnergyBasedViscosity<Part,
    mini::riemann::ConvectionDiffusion<Riemann, Diffusion>>;
static_assert(mini::riemann::ConvectiveDiffusive<RiemannWithViscosity>);

#if defined(FR)
#include "mini/spatial/fr/lobatto.hpp"
using General = mini::spatial::fr::Lobatto<Part, RiemannWithViscosity>;
#endif

using Spatial = mini::spatial::WithViscosity<General>;

#endif  // VISCOSITY

/* Choose the time-stepping scheme. */
#include "mini/temporal/rk.hpp"
constexpr int kOrders = std::min(3, kDegrees + 1);
using Temporal = mini::temporal::RungeKutta<kOrders, Scalar>;

/* Define the types of IC and BCs. */
using IC = Value(*)(const Global &);
using BC = void(*)(const std::string &, Spatial *);

int Main(int argc, char* argv[], IC ic, BC bc);

#endif  // DEMO_EULER_SOURCELESS_HPP_
