//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/integrator/function.hpp"
#include "mini/integrator/quadrangle.hpp"
#include "mini/coordinate/quadrangle.hpp"
#include "mini/basis/linear.hpp"
#include "mini/polynomial/projection.hpp"

#include "gtest/gtest.h"

using std::sqrt;

class TestQuadrangle4x4 : public ::testing::Test {
};
TEST_F(TestQuadrangle4x4, OrthoNormal) {
  using Basis = mini::basis::OrthoNormal<double, 2, 2>;
  using Gx = mini::integrator::Legendre<double, 4>;
  using Integrator = mini::integrator::Quadrangle<2, Gx, Gx>;
  using Coordinate = mini::coordinate::Quadrangle4<double, 2>;
  using Coord = typename Coordinate::Global;
  Coord origin = {0, 0}, left = {-1, 2}, right = {1, 3};
  auto coordinate = Coordinate {
    Coord(-1, -1), Coord(1, -1), Coord(1, 1), Coord(-1, 1)
  };
  auto integrator = Integrator(coordinate);
  auto basis = Basis(integrator);
  using A = typename Basis::MatNxN;
  double residual = (Integrate([&basis](const Coord& xy) {
    auto col = basis(xy);
    A prod = col * col.transpose();
    return prod;
  }, integrator) - A::Identity()).norm();
  EXPECT_NEAR(residual, 0.0, 1e-14);
  auto x = left[0], y = left[1];
  coordinate = Coordinate {
    Coord(x-1, y-1), Coord(x+1, y-1), Coord(x+1, y+1), Coord(x-1, y+1)
  };
  basis = Basis(integrator);
  residual = (Integrate([&basis](Coord const& xy) {
    auto col = basis(xy);
    A prod = col * col.transpose();
    return prod;
  }, integrator) - A::Identity()).norm();
  EXPECT_NEAR(residual, 0.0, 1e-12);
}
TEST_F(TestQuadrangle4x4, Projection) {
  using Basis = mini::basis::OrthoNormal<double, 2, 2>;
  using Gx = mini::integrator::Legendre<double, 4>;
  using Integrator = mini::integrator::Quadrangle<2, Gx, Gx>;
  using Coordinate = mini::coordinate::Quadrangle4<double, 2>;
  using Coord = typename Coordinate::Global;
  auto coordinate = Coordinate {
    Coord(-1, -1), Coord(1, -1), Coord(1, 1), Coord(-1, 1)
  };
  auto integrator = Integrator(coordinate);
  auto scalar_f = [](Coord const& xy){
    return xy[0] * xy[1];
  };
  using ScalarPF = mini::polynomial::Projection<double, 2, 2, 1>;
  auto scalar_pf = ScalarPF(integrator);
  scalar_pf.Approximate(scalar_f);
  using Mat1x6 = mini::algebra::Matrix<double, 1, 6>;
  double residual = (scalar_pf.GetCoeffOnTaylorBasis()
      - Mat1x6(0, 0, 0, 0, 1, 0)).norm();
  EXPECT_NEAR(residual, 0.0, 1e-15);
  using Mat7x1 = mini::algebra::Matrix<double, 7, 1>;
  auto vector_f = [](Coord const& xy) {
    auto x = xy[0], y = xy[1];
    Mat7x1 func(0, 1, x, y, x * x, x * y, y * y);
    return func;
  };
  using VectorPF = mini::polynomial::Projection<double, 2, 2, 7>;
  auto vector_pf = VectorPF(integrator);
  vector_pf.Approximate(vector_f);
  using Mat7x6 = mini::algebra::Matrix<double, 7, 6>;
  Mat7x6 exact_vector{
      {0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0},
      {0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0},
      {0, 0, 0, 0, 0, 1}
  };
  Mat7x6 abs_diff = vector_pf.GetCoeffOnTaylorBasis() - exact_vector;
  EXPECT_NEAR(abs_diff.norm(), 0.0, 1e-14);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
