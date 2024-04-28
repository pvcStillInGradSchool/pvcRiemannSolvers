//  Copyright 2022 PEI Weicheng

#include <cmath>

#include "mini/integrator/function.hpp"
#include "mini/integrator/tetrahedron.hpp"
#include "mini/coordinate/tetrahedron.hpp"
#include "mini/basis/linear.hpp"
#include "mini/polynomial/projection.hpp"

#include "gtest/gtest.h"

using std::sqrt;

class TestTetrahedron : public ::testing::Test {
 protected:
  using Integrator = mini::integrator::Tetrahedron<double, 14>;
  using Coordinate = mini::coordinate::Tetrahedron4<double>;
  using Basis = mini::basis::OrthoNormal<double, 3, 2>;
  using Coord = typename Basis::Coord;
  using A = typename Basis::MatNxN;
};
TEST_F(TestTetrahedron, OrthoNormal) {
  // build a tetra-gauss
  auto coordinate = Coordinate {
    Coord(10, 10, 0), Coord(0, 10, 10),
    Coord(10, 0, 10), Coord(10, 10, 10)
  };
  auto tetra = Integrator(coordinate);
  // build an orthonormal basis on it
  auto basis = Basis(tetra);
  // check orthonormality
  double residual = (Integrate([&basis](const Coord& xyz) {
    auto col = basis(xyz);
    A prod = col * col.transpose();
    return prod;
  }, tetra) - A::Identity()).norm();
  EXPECT_NEAR(residual, 0.0, 1e-14);
  // build another tetra-gauss
  Coord shift = {10, 20, 30};
  coordinate = Coordinate {
    coordinate.GetGlobal(0) + shift,
    coordinate.GetGlobal(1) + shift,
    coordinate.GetGlobal(2) + shift,
    coordinate.GetGlobal(3) + shift
  };
  tetra = Integrator(coordinate);
  // build another orthonormal basis on it
  basis = Basis(tetra);
  // check orthonormality
  residual = (Integrate([&basis](const Coord& xyz) {
    auto col = basis(xyz);
    A prod = col * col.transpose();
    return prod;
  }, tetra) - A::Identity()).norm();
  EXPECT_NEAR(residual, 0.0, 1e-14);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
