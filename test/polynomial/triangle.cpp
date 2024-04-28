//  Copyright 2022 PEI Weicheng

#include <cmath>

#include "mini/integrator/function.hpp"
#include "mini/integrator/triangle.hpp"
#include "mini/coordinate/triangle.hpp"
#include "mini/basis/linear.hpp"
#include "mini/polynomial/projection.hpp"

#include "gtest/gtest.h"

using std::sqrt;

class TestTriangle : public ::testing::Test {
};
TEST_F(TestTriangle, OrthoNormal) {
  using Basis = mini::basis::OrthoNormal<double, 2, 2>;
  using Integrator = mini::integrator::Triangle<double, 2, 12>;
  using Coordinate = mini::coordinate::Triangle3<double, 2>;
  using Coord = typename Coordinate::Global;
  // build a triangle-integrator
  auto coordinate = Coordinate { Coord(10, 0), Coord(0, 10), Coord(0, 0) };
  auto integrator = Integrator(coordinate);
  // build an orthonormal basis on it
  auto basis = Basis(integrator);
  // check orthonormality
  using A = typename Basis::MatNxN;
  double residual = (Integrate([&basis](const Coord& xyz) {
    auto col = basis(xyz);
    A prod = col * col.transpose();
    return prod;
  }, integrator) - A::Identity()).norm();
  EXPECT_NEAR(residual, 0.0, 1e-14);
  // build another triangle-integrator
  Coord shift = {10, 20};
  coordinate = Coordinate {
    Coord(10, 0) + shift, Coord(0, 10) + shift, Coord(0, 0) + shift
  };
  integrator = Integrator(coordinate);
  // build another orthonormal basis on it
  basis = Basis(integrator);
  // check orthonormality
  residual = (Integrate([&basis](const Coord& xyz) {
    auto col = basis(xyz);
    A prod = col * col.transpose();
    return prod;
  }, integrator) - A::Identity()).norm();
  EXPECT_NEAR(residual, 0.0, 1e-14);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
