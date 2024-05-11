// Copyright 2024 PEI Weicheng
#ifndef TEST_MESH_PART_HPP_
#define TEST_MESH_PART_HPP_

#include "mini/riemann/concept.hpp"
#include "mini/riemann/rotated/multiple.hpp"
#include "mini/riemann/diffusive/linear.hpp"
#include "mini/riemann/diffusive/direct_dg.hpp"
#include "mini/integrator/lobatto.hpp"
#include "mini/coordinate/quadrangle.hpp"
#include "mini/integrator/quadrangle.hpp"

#include "mpi.h"
#include "gtest/gtest.h"
#include "gtest_mpi/gtest_mpi.hpp"

constexpr int kComponents{2}, kDimensions{3}, kDegrees{2};

using Scalar = double;
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
  Riemann::SetDiffusionCoefficient(1.0);
  Riemann::SetBetaValues(2.0, 1.0 / 12);
}

using Coord = typename Riemann::Vector;
using Value = typename Riemann::Conservative;

Value func(const Coord& xyz) {
  auto r = std::hypot(xyz[0] - 2, xyz[1] - 0.5);
  return Value(r, 1 - r + (r >= 1));
}

Value moving(const Coord& xyz, double t) {
  auto x = xyz[0], y = xyz[1];
  return Value(x + y, x - y);
}

int n_core, i_core;
double time_begin;

using Gx = mini::integrator::Lobatto<Scalar, kDegrees + 1>;
using QuadrangleIntegrator
    = mini::integrator::Quadrangle<kDimensions, Gx, Gx>;

int Main(int argc, char* argv[]) {
  // Initialize MPI before any call to gtest_mpi
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &n_core);
  MPI_Comm_rank(MPI_COMM_WORLD, &i_core);
  cgp_mpi_comm(MPI_COMM_WORLD);

  // Intialize google test
  ::testing::InitGoogleTest(&argc, argv);

  // Add a test environment, which will initialize a test communicator
  // (a duplicate of MPI_COMM_WORLD)
  ::testing::AddGlobalTestEnvironment(new gtest_mpi::MPITestEnvironment());

  auto& test_listeners = ::testing::UnitTest::GetInstance()->listeners();

  // Remove default listener and replace with the custom MPI listener
  delete test_listeners.Release(test_listeners.default_result_printer());
  test_listeners.Append(new gtest_mpi::PrettyMPIUnitTestResultPrinter());

  // run tests
  auto exit_code = RUN_ALL_TESTS();

  // Finalize MPI before exiting
  MPI_Finalize();

  return exit_code;
}

#endif  // TEST_MESH_PART_HPP_
