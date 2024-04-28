//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/integrator/function.hpp"
#include "mini/integrator/triangle.hpp"
#include "mini/coordinate/triangle.hpp"

#include "gtest/gtest.h"

class TestIntegratorTriangle : public ::testing::Test {
};
TEST_F(TestIntegratorTriangle, OnScaledElementInTwoDimensionalSpace) {
  using Integrator = mini::integrator::Triangle<double, 2, 16>;
  using Coordinate = mini::coordinate::Triangle3<double, 2>;
  using Coord = typename Coordinate::Global;
  auto coordinate = Coordinate { Coord(0, 0), Coord(2, 0), Coord(2, 2) };
  auto integrator = Integrator(coordinate);
  EXPECT_EQ(integrator.CountPoints(), 16);
  static_assert(integrator.CellDim() == 2);
  static_assert(integrator.PhysDim() == 2);
  EXPECT_DOUBLE_EQ(integrator.area(), 2.0);
  EXPECT_NEAR((integrator.center() - Coord(4./3, 2./3)).norm(), 0, 1e-15);
  EXPECT_DOUBLE_EQ(Quadrature([](Coord const&){ return 2.0; }, integrator), 1.0);
  EXPECT_DOUBLE_EQ(Integrate([](Coord const&){ return 2.0; }, integrator), 4.0);
  auto f = [](Coord const& xy){ return xy[0]; };
  auto g = [](Coord const& xy){ return xy[1]; };
  auto h = [](Coord const& xy){ return xy[0] * xy[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, integrator), Integrate(h, integrator));
  EXPECT_DOUBLE_EQ(Norm(f, integrator), std::sqrt(Innerprod(f, f, integrator)));
  EXPECT_DOUBLE_EQ(Norm(g, integrator), std::sqrt(Innerprod(g, g, integrator)));
}
TEST_F(TestIntegratorTriangle, OnMappedElementInThreeDimensionalSpace) {
  using Integrator = mini::integrator::Triangle<double, 3, 16>;
  using Coordinate = mini::coordinate::Triangle3<double, 3>;
  using Local = typename Coordinate::Local;
  using Global = typename Coordinate::Global;
  auto coordinate = Coordinate {
    Global(0, 0, 2), Global(2, 0, 2), Global(2, 2, 2),
  };
  auto const integrator = Integrator(coordinate);
  static_assert(integrator.CellDim() == 2);
  static_assert(integrator.PhysDim() == 3);
  EXPECT_DOUBLE_EQ(integrator.area(), 2.0);
  EXPECT_NEAR((integrator.center() - Global(4./3, 2./3, 2.)).norm(), 0, 1e-15);
  EXPECT_DOUBLE_EQ(
      Quadrature([](Local const&){ return 2.0; }, integrator), 1.0);
  EXPECT_DOUBLE_EQ(
      Integrate([](Global const&){ return 2.0; }, integrator), 4.0);
  auto f = [](Global const& xyz){ return xyz[0]; };
  auto g = [](Global const& xyz){ return xyz[1]; };
  auto h = [](Global const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, integrator), Integrate(h, integrator));
  EXPECT_DOUBLE_EQ(Norm(f, integrator), std::sqrt(Innerprod(f, f, integrator)));
  EXPECT_DOUBLE_EQ(Norm(g, integrator), std::sqrt(Innerprod(g, g, integrator)));
  // test normal frames
  Global normal = Global(0, 0, 1).normalized();
  for (int q = 0; q < integrator.CountPoints(); ++q) {
    auto &frame = integrator.GetNormalFrame(q);
    auto &nu = frame[0], &sigma = frame[1], &pi = frame[2];
    EXPECT_NEAR((nu - normal).norm(), 0.0, 1e-15);
    EXPECT_NEAR((nu - sigma.cross(pi)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((sigma - pi.cross(nu)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((pi - nu.cross(sigma)).norm(), 0.0, 1e-15);
  }
}
TEST_F(TestIntegratorTriangle, OnQuadraticElementInThreeDimensionalSpace) {
  using Integrator = mini::integrator::Triangle<double, 3, 16>;
  using Coordinate = mini::coordinate::Triangle6<double, 3>;
  using Local = typename Coordinate::Local;
  using Global = typename Coordinate::Global;
  auto coordinate = Coordinate {
    Global(0, 0, 2), Global(2, 0, 2), Global(2, 2, 2),
    Global(1, 0, 2), Global(2, 1, 2), Global(1, 1, 2),
  };
  auto const integrator = Integrator(coordinate);
  static_assert(integrator.CellDim() == 2);
  static_assert(integrator.PhysDim() == 3);
  EXPECT_DOUBLE_EQ(integrator.area(), 2.0);
  EXPECT_NEAR((integrator.center() - Global(4./3, 2./3, 2.)).norm(), 0, 1e-15);
  EXPECT_DOUBLE_EQ(
      Quadrature([](Local const&){ return 2.0; }, integrator), 1.0);
  EXPECT_DOUBLE_EQ(
      Integrate([](Global const&){ return 2.0; }, integrator), 4.0);
  auto f = [](Global const& xyz){ return xyz[0]; };
  auto g = [](Global const& xyz){ return xyz[1]; };
  auto h = [](Global const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, integrator), Integrate(h, integrator));
  EXPECT_DOUBLE_EQ(Norm(f, integrator), std::sqrt(Innerprod(f, f, integrator)));
  EXPECT_DOUBLE_EQ(Norm(g, integrator), std::sqrt(Innerprod(g, g, integrator)));
  // test normal frames
  Global normal = Global(0, 0, 1).normalized();
  for (int q = 0; q < integrator.CountPoints(); ++q) {
    auto &frame = integrator.GetNormalFrame(q);
    auto &nu = frame[0], &sigma = frame[1], &pi = frame[2];
    EXPECT_NEAR((nu - normal).norm(), 0.0, 1e-15);
    EXPECT_NEAR((nu - sigma.cross(pi)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((sigma - pi.cross(nu)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((pi - nu.cross(sigma)).norm(), 0.0, 1e-15);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
