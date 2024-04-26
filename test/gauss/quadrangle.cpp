//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/gauss/function.hpp"
#include "mini/gauss/quadrangle.hpp"
#include "mini/geometry/quadrangle.hpp"

#include "gtest/gtest.h"

class TestGaussQuadrangle : public ::testing::Test {
};
TEST_F(TestGaussQuadrangle, TwoDimensionalQuadrangle4) {
  using Gx = mini::gauss::Legendre<double, 4>;
  using Gauss = mini::gauss::Quadrangle<2, Gx, Gx>;
  using Coordinate = mini::geometry::Quadrangle4<double, 2>;
  using Coord = typename Coordinate::Global;
  auto coordinate = Coordinate {
    Coord(-1, -1), Coord(1, -1), Coord(1, 1), Coord(-1, 1)
  };
  auto gauss = Gauss(coordinate);
  static_assert(gauss.CellDim() == 2);
  static_assert(gauss.PhysDim() == 2);
  EXPECT_EQ(gauss.CountPoints(), 16);
  auto p0 = gauss.GetLocalCoord(0);
  EXPECT_EQ(p0[0], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  EXPECT_EQ(p0[1], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  auto w1d = (18 - std::sqrt(30)) / 36.0;
  EXPECT_EQ(gauss.GetLocalWeight(0), w1d * w1d);
  EXPECT_NEAR(gauss.area(), 4.0, 1e-15);
  EXPECT_DOUBLE_EQ(Quadrature([](Coord const&){ return 2.0; }, gauss), 8.0);
  EXPECT_DOUBLE_EQ(Integrate([](Coord const&){ return 2.0; }, gauss), 8.0);
  auto f = [](Coord const& xy){ return xy[0]; };
  auto g = [](Coord const& xy){ return xy[1]; };
  auto h = [](Coord const& xy){ return xy[0] * xy[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, gauss), Integrate(h, gauss));
  EXPECT_DOUBLE_EQ(Norm(f, gauss), std::sqrt(Innerprod(f, f, gauss)));
  EXPECT_DOUBLE_EQ(Norm(g, gauss), std::sqrt(Innerprod(g, g, gauss)));
}
TEST_F(TestGaussQuadrangle, ThreeDimensionalQuadrangle4) {
  using Gx = mini::gauss::Legendre<double, 4>;
  using Gauss = mini::gauss::Quadrangle<3, Gx, Gx>;
  using Coordinate = mini::geometry::Quadrangle4<double, 3>;
  using Local = typename Coordinate::Local;
  using Global = typename Coordinate::Global;
  auto coordinate = Coordinate {
    Global(0, 0, 0), Global(4, 0, 0), Global(4, 4, 4), Global(0, 4, 4)
  };
  auto const gauss = Gauss(coordinate);
  static_assert(gauss.CellDim() == 2);
  static_assert(gauss.PhysDim() == 3);
  EXPECT_NEAR(gauss.area(), sqrt(2) * 16.0, 1e-14);
  EXPECT_DOUBLE_EQ(
      Quadrature([](Local const&){ return 2.0; }, gauss), 8.0);
  EXPECT_DOUBLE_EQ(
      Integrate([](Global const&){ return 2.0; }, gauss), sqrt(2) * 32.0);
  auto f = [](Global const& xyz){ return xyz[0]; };
  auto g = [](Global const& xyz){ return xyz[1]; };
  auto h = [](Global const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, gauss), Integrate(h, gauss));
  EXPECT_DOUBLE_EQ(Norm(f, gauss), std::sqrt(Innerprod(f, f, gauss)));
  EXPECT_DOUBLE_EQ(Norm(g, gauss), std::sqrt(Innerprod(g, g, gauss)));
  // test normal frames
  Global normal = Global(0, -1, 1).normalized();
  for (int q = 0; q < gauss.CountPoints(); ++q) {
    auto &frame = gauss.GetNormalFrame(q);
    auto &nu = frame[0], &sigma = frame[1], &pi = frame[2];
    EXPECT_NEAR((nu - normal).norm(), 0.0, 1e-15);
    EXPECT_NEAR((nu - sigma.cross(pi)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((sigma - pi.cross(nu)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((pi - nu.cross(sigma)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((sigma - Global(1, 0, 0)).norm(), 0.0, 1e-15);
  }
}
TEST_F(TestGaussQuadrangle, ThreeDimensionalQuadrangle8) {
  using Gx = mini::gauss::Legendre<double, 4>;
  using Gauss = mini::gauss::Quadrangle<3, Gx, Gx>;
  using Coordinate = mini::geometry::Quadrangle8<double, 3>;
  using Local = typename Coordinate::Local;
  using Global = typename Coordinate::Global;
  auto coordinate = Coordinate {
    Global(0, 0, 0), Global(4, 0, 0), Global(4, 4, 4), Global(0, 4, 4),
    Global(2, 0, 0), Global(4, 2, 2), Global(2, 4, 4), Global(0, 2, 2),
  };
  auto const gauss = Gauss(coordinate);
  static_assert(gauss.CellDim() == 2);
  static_assert(gauss.PhysDim() == 3);
  EXPECT_NEAR(gauss.area(), sqrt(2) * 16.0, 1e-14);
  EXPECT_DOUBLE_EQ(
      Quadrature([](Local const&){ return 2.0; }, gauss), 8.0);
  EXPECT_DOUBLE_EQ(
      Integrate([](Global const&){ return 2.0; }, gauss), sqrt(2) * 32.0);
  auto f = [](Global const& xyz){ return xyz[0]; };
  auto g = [](Global const& xyz){ return xyz[1]; };
  auto h = [](Global const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, gauss), Integrate(h, gauss));
  EXPECT_DOUBLE_EQ(Norm(f, gauss), std::sqrt(Innerprod(f, f, gauss)));
  EXPECT_DOUBLE_EQ(Norm(g, gauss), std::sqrt(Innerprod(g, g, gauss)));
  // test normal frames
  Global normal = Global(0, -1, 1).normalized();
  for (int q = 0; q < gauss.CountPoints(); ++q) {
    auto &frame = gauss.GetNormalFrame(q);
    auto &nu = frame[0], &sigma = frame[1], &pi = frame[2];
    EXPECT_NEAR((nu - normal).norm(), 0.0, 1e-15);
    EXPECT_NEAR((nu - sigma.cross(pi)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((sigma - pi.cross(nu)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((pi - nu.cross(sigma)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((sigma - Global(1, 0, 0)).norm(), 0.0, 1e-15);
  }
}
TEST_F(TestGaussQuadrangle, ThreeDimensionalQuadrangle9) {
  using Gx = mini::gauss::Legendre<double, 4>;
  using Gauss = mini::gauss::Quadrangle<3, Gx, Gx>;
  using Coordinate = mini::geometry::Quadrangle9<double, 3>;
  using Local = typename Coordinate::Local;
  using Global = typename Coordinate::Global;
  auto coordinate = Coordinate {
    Global(0, 0, 0), Global(4, 0, 0), Global(4, 4, 4), Global(0, 4, 4),
    Global(2, 0, 0), Global(4, 2, 2), Global(2, 4, 4), Global(0, 2, 2),
    Global(2, 2, 2),
  };
  auto const gauss = Gauss(coordinate);
  static_assert(gauss.CellDim() == 2);
  static_assert(gauss.PhysDim() == 3);
  EXPECT_NEAR(gauss.area(), sqrt(2) * 16.0, 1e-14);
  EXPECT_DOUBLE_EQ(
      Quadrature([](Local const&){ return 2.0; }, gauss), 8.0);
  EXPECT_DOUBLE_EQ(
      Integrate([](Global const&){ return 2.0; }, gauss), sqrt(2) * 32.0);
  auto f = [](Global const& xyz){ return xyz[0]; };
  auto g = [](Global const& xyz){ return xyz[1]; };
  auto h = [](Global const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, gauss), Integrate(h, gauss));
  EXPECT_DOUBLE_EQ(Norm(f, gauss), std::sqrt(Innerprod(f, f, gauss)));
  EXPECT_DOUBLE_EQ(Norm(g, gauss), std::sqrt(Innerprod(g, g, gauss)));
  // test normal frames
  Global normal = Global(0, -1, 1).normalized();
  for (int q = 0; q < gauss.CountPoints(); ++q) {
    auto &frame = gauss.GetNormalFrame(q);
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
