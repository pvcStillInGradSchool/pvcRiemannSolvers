// Copyright 2024 PEI Weicheng
#ifndef TEST_MESH_PART_HPP_
#define TEST_MESH_PART_HPP_

#include <memory>

#include "mini/algebra/eigen.hpp"
#include "mini/integrator/lobatto.hpp"
#include "mini/coordinate/quadrangle.hpp"
#include "mini/coordinate/hexahedron.hpp"
#include "mini/integrator/quadrangle.hpp"
#include "mini/integrator/hexahedron.hpp"

#include "mpi.h"
#include "gtest/gtest.h"
#include "gtest_mpi/gtest_mpi.hpp"

constexpr int kComponents{2}, kDimensions{3}, kDegrees{2};

using Scalar = double;
using Value = mini::algebra::Vector<Scalar, kComponents>;
using Coord = mini::algebra::Vector<Scalar, kDimensions>;

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
using HexahedronIntegrator
    = mini::integrator::Hexahedron<Gx, Gx, Gx>;

template <class Part>
void InstallIntegratorPrototypes(Part *part_ptr) {
  auto quadrangle = mini::coordinate::Quadrangle4<Scalar, kDimensions>();
  part_ptr->InstallPrototype(4,
      std::make_unique<QuadrangleIntegrator>(quadrangle));
  auto hexahedron = mini::coordinate::Hexahedron8<Scalar>();
  part_ptr->InstallPrototype(8,
      std::make_unique<HexahedronIntegrator>(hexahedron));
  part_ptr->BuildGeometry();
}

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
