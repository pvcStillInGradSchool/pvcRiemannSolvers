//  Copyright 2022 PEI Weicheng
#ifndef DEMO_EULER_ROTORCRAFT_HPP_
#define DEMO_EULER_ROTORCRAFT_HPP_

#include <algorithm>
#include <string>

#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/eigen.hpp"
#include "mini/riemann/euler/exact.hpp"
#include "mini/riemann/rotated/euler.hpp"
#include "mini/polynomial/projection.hpp"
#include "mini/mesh/part.hpp"
#include "mini/limiter/weno.hpp"
#include "mini/limiter/reconstruct.hpp"
#include "mini/temporal/rk.hpp"
#include "mini/spatial/dg/general.hpp"
#include "mini/aircraft/source.hpp"

using Scalar = double;

/* Define the Euler system. */
constexpr int kDimensions = 3;
using Primitive = mini::riemann::euler::Primitives<Scalar, kDimensions>;
using Conservative = mini::riemann::euler::Conservatives<Scalar, kDimensions>;
using Gas = mini::riemann::euler::IdealGas<Scalar, 1.4>;
using Unrotated = mini::riemann::euler::Exact<Gas, kDimensions>;
using Riemann = mini::riemann::rotated::Euler<Unrotated>;

/* Define spatial discretization. */
constexpr int kDegrees = 2;
using Projection = mini::polynomial::Projection<double, kDimensions, kDegrees, 5>;
using Part = mini::mesh::part::Part<cgsize_t, Riemann, Projection>;
using Cell = typename Part::Cell;
using Face = typename Part::Face;
using Global = typename Cell::Global;
using Value = typename Cell::Value;
using Coeff = typename Cell::Coeff;

using Limiter = mini::limiter::weno::Eigen<Cell>;

using Source = mini::aircraft::Rotorcraft<Part, Scalar>;
using Rotor = mini::aircraft::Rotor<Scalar>;
using Blade = typename Rotor::Blade;
using Frame = typename Blade::Frame;
using Airfoil = typename Blade::Airfoil;

using Spatial = mini::spatial::dg::WithLimiterAndSource<Part, Limiter, Source>;

/* Choose the time-stepping scheme. */
constexpr int kOrders = std::min(3, kDegrees + 1);
using Temporal = mini::temporal::RungeKutta<kOrders, Scalar>;

using IC = Value(*)(const Global &);
using BC = void(*)(const std::string &, Spatial *);

void WriteForces(Part const &part, Source *source, double t_curr,
    std::string const &frame_name, int i_core);

int Main(int argc, char* argv[], IC ic, BC bc, Source source);

#endif  // DEMO_EULER_ROTORCRAFT_HPP_
