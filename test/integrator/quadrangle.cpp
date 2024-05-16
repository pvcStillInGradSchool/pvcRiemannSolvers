//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/integrator/function.hpp"
#include "mini/integrator/legendre.hpp"
#include "mini/integrator/quadrangle.hpp"
#include "mini/coordinate/quadrangle.hpp"

#include "gtest/gtest.h"

class TestIntegratorQuadrangle : public ::testing::Test {
};
TEST_F(TestIntegratorQuadrangle, TwoDimensionalQuadrangle4) {
  using Gx = mini::integrator::Legendre<double, 4>;
  using Integrator = mini::integrator::Quadrangle<2, Gx, Gx>;
  using Coordinate = mini::coordinate::Quadrangle4<double, 2>;
  using Coord = typename Coordinate::Global;
  auto coordinate = Coordinate {
    Coord(-1, -1), Coord(1, -1), Coord(1, 1), Coord(-1, 1)
  };
  auto integrator = Integrator(coordinate);
  static_assert(integrator.CellDim() == 2);
  static_assert(integrator.PhysDim() == 2);
  EXPECT_EQ(integrator.CountPoints(), 16);
  auto p0 = integrator.GetLocal(0);
  EXPECT_EQ(p0[0], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  EXPECT_EQ(p0[1], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  auto w1d = (18 - std::sqrt(30)) / 36.0;
  EXPECT_EQ(integrator.GetLocalWeight(0), w1d * w1d);
  EXPECT_NEAR(integrator.area(), 4.0, 1e-15);
  EXPECT_DOUBLE_EQ(Quadrature([](Coord const&){ return 2.0; }, integrator), 8.0);
  EXPECT_DOUBLE_EQ(Integrate([](Coord const&){ return 2.0; }, integrator), 8.0);
  auto f = [](Coord const& xy){ return xy[0]; };
  auto g = [](Coord const& xy){ return xy[1]; };
  auto h = [](Coord const& xy){ return xy[0] * xy[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, integrator), Integrate(h, integrator));
  EXPECT_DOUBLE_EQ(Norm(f, integrator), std::sqrt(Innerprod(f, f, integrator)));
  EXPECT_DOUBLE_EQ(Norm(g, integrator), std::sqrt(Innerprod(g, g, integrator)));
}
TEST_F(TestIntegratorQuadrangle, ThreeDimensionalQuadrangle4) {
  using Gx = mini::integrator::Legendre<double, 4>;
  using Integrator = mini::integrator::Quadrangle<3, Gx, Gx>;
  using Coordinate = mini::coordinate::Quadrangle4<double, 3>;
  using Local = typename Coordinate::Local;
  using Global = typename Coordinate::Global;
  auto coordinate = Coordinate {
    Global(0, 0, 0), Global(4, 0, 0), Global(4, 4, 4), Global(0, 4, 4)
  };
  auto const integrator = Integrator(coordinate);
  static_assert(integrator.CellDim() == 2);
  static_assert(integrator.PhysDim() == 3);
  EXPECT_NEAR(integrator.area(), sqrt(2) * 16.0, 1e-14);
  EXPECT_DOUBLE_EQ(
      Quadrature([](Local const&){ return 2.0; }, integrator), 8.0);
  EXPECT_DOUBLE_EQ(
      Integrate([](Global const&){ return 2.0; }, integrator), sqrt(2) * 32.0);
  auto f = [](Global const& xyz){ return xyz[0]; };
  auto g = [](Global const& xyz){ return xyz[1]; };
  auto h = [](Global const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, integrator), Integrate(h, integrator));
  EXPECT_DOUBLE_EQ(Norm(f, integrator), std::sqrt(Innerprod(f, f, integrator)));
  EXPECT_DOUBLE_EQ(Norm(g, integrator), std::sqrt(Innerprod(g, g, integrator)));
  // test normal frames
  Global normal = Global(0, -1, 1).normalized();
  for (int q = 0; q < integrator.CountPoints(); ++q) {
    auto &frame = integrator.GetNormalFrame(q);
    auto &nu = frame[0], &sigma = frame[1], &pi = frame[2];
    EXPECT_NEAR((nu - normal).norm(), 0.0, 1e-15);
    EXPECT_NEAR((nu - sigma.cross(pi)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((sigma - pi.cross(nu)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((pi - nu.cross(sigma)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((sigma - Global(1, 0, 0)).norm(), 0.0, 1e-15);
  }
}
TEST_F(TestIntegratorQuadrangle, ThreeDimensionalQuadrangle8) {
  using Gx = mini::integrator::Legendre<double, 4>;
  using Integrator = mini::integrator::Quadrangle<3, Gx, Gx>;
  using Coordinate = mini::coordinate::Quadrangle8<double, 3>;
  using Local = typename Coordinate::Local;
  using Global = typename Coordinate::Global;
  auto coordinate = Coordinate {
    Global(0, 0, 0), Global(4, 0, 0), Global(4, 4, 4), Global(0, 4, 4),
    Global(2, 0, 0), Global(4, 2, 2), Global(2, 4, 4), Global(0, 2, 2),
  };
  auto const integrator = Integrator(coordinate);
  static_assert(integrator.CellDim() == 2);
  static_assert(integrator.PhysDim() == 3);
  EXPECT_NEAR(integrator.area(), sqrt(2) * 16.0, 1e-14);
  EXPECT_DOUBLE_EQ(
      Quadrature([](Local const&){ return 2.0; }, integrator), 8.0);
  EXPECT_DOUBLE_EQ(
      Integrate([](Global const&){ return 2.0; }, integrator), sqrt(2) * 32.0);
  auto f = [](Global const& xyz){ return xyz[0]; };
  auto g = [](Global const& xyz){ return xyz[1]; };
  auto h = [](Global const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, integrator), Integrate(h, integrator));
  EXPECT_DOUBLE_EQ(Norm(f, integrator), std::sqrt(Innerprod(f, f, integrator)));
  EXPECT_DOUBLE_EQ(Norm(g, integrator), std::sqrt(Innerprod(g, g, integrator)));
  // test normal frames
  Global normal = Global(0, -1, 1).normalized();
  for (int q = 0; q < integrator.CountPoints(); ++q) {
    auto &frame = integrator.GetNormalFrame(q);
    auto &nu = frame[0], &sigma = frame[1], &pi = frame[2];
    EXPECT_NEAR((nu - normal).norm(), 0.0, 1e-15);
    EXPECT_NEAR((nu - sigma.cross(pi)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((sigma - pi.cross(nu)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((pi - nu.cross(sigma)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((sigma - Global(1, 0, 0)).norm(), 0.0, 1e-15);
  }
}
TEST_F(TestIntegratorQuadrangle, ThreeDimensionalQuadrangle9) {
  using Gx = mini::integrator::Legendre<double, 4>;
  using Integrator = mini::integrator::Quadrangle<3, Gx, Gx>;
  using Coordinate = mini::coordinate::Quadrangle9<double, 3>;
  using Local = typename Coordinate::Local;
  using Global = typename Coordinate::Global;
  auto coordinate = Coordinate {
    Global(0, 0, 0), Global(4, 0, 0), Global(4, 4, 4), Global(0, 4, 4),
    Global(2, 0, 0), Global(4, 2, 2), Global(2, 4, 4), Global(0, 2, 2),
    Global(2, 2, 2),
  };
  auto const integrator = Integrator(coordinate);
  static_assert(integrator.CellDim() == 2);
  static_assert(integrator.PhysDim() == 3);
  EXPECT_NEAR(integrator.area(), sqrt(2) * 16.0, 1e-14);
  EXPECT_DOUBLE_EQ(
      Quadrature([](Local const&){ return 2.0; }, integrator), 8.0);
  EXPECT_DOUBLE_EQ(
      Integrate([](Global const&){ return 2.0; }, integrator), sqrt(2) * 32.0);
  auto f = [](Global const& xyz){ return xyz[0]; };
  auto g = [](Global const& xyz){ return xyz[1]; };
  auto h = [](Global const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, integrator), Integrate(h, integrator));
  EXPECT_DOUBLE_EQ(Norm(f, integrator), std::sqrt(Innerprod(f, f, integrator)));
  EXPECT_DOUBLE_EQ(Norm(g, integrator), std::sqrt(Innerprod(g, g, integrator)));
  // test normal frames
  Global normal = Global(0, -1, 1).normalized();
  for (int q = 0; q < integrator.CountPoints(); ++q) {
    auto &frame = integrator.GetNormalFrame(q);
    auto &nu = frame[0], &sigma = frame[1], &pi = frame[2];
    EXPECT_NEAR((nu - normal).norm(), 0.0, 1e-15);
    EXPECT_NEAR((nu - sigma.cross(pi)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((sigma - pi.cross(nu)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((pi - nu.cross(sigma)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((sigma - Global(1, 0, 0)).norm(), 0.0, 1e-15);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
