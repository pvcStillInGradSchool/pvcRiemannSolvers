//  Copyright 2023 PEI Weicheng

#include <cmath>

#include "mini/integrator/function.hpp"
#include "mini/integrator/pyramid.hpp"
#include "mini/coordinate/pyramid.hpp"

#include "gtest/gtest.h"

class TestGaussPyramid : public ::testing::Test {
};
TEST_F(TestGaussPyramid, OnLinearElement) {
  using Coordinate = mini::coordinate::Pyramid5<double>;
  using Gauss = mini::integrator::Pyramid<double, 4, 4, 3>;
  using Coord = typename Gauss::Global;
  auto a = 2.0, b = 3.0, h = 4.0;
  auto coordinate = Coordinate{
    Coord(-a, -b, 0), Coord(+a, -b, 0),
    Coord(+a, +b, 0), Coord(-a, +b, 0),
    Coord(0, 0, h)
  };
  auto gauss = Gauss(coordinate);
  static_assert(gauss.CellDim() == 3);
  static_assert(gauss.PhysDim() == 3);
  auto volume = (a + a) * (b + b) * h / 3;
  EXPECT_NEAR(gauss.volume(), volume, 1e-13);
  EXPECT_EQ(gauss.CountPoints(), 48);
  EXPECT_NEAR(Integrate([](Coord const&){ return 2.0; }, gauss),
      volume * 2, 1e-13);
  auto f = [](Coord const& xyz){ return xyz[0]; };
  auto g = [](Coord const& xyz){ return xyz[1]; };
  auto fg = [](Coord const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, gauss), Integrate(fg, gauss));
  EXPECT_DOUBLE_EQ(Norm(f, gauss), std::sqrt(Innerprod(f, f, gauss)));
  EXPECT_DOUBLE_EQ(Norm(g, gauss), std::sqrt(Innerprod(g, g, gauss)));
}
TEST_F(TestGaussPyramid, OnQuadraticElement) {
  using Coordinate = mini::coordinate::Pyramid13<double>;
  using Gauss = mini::integrator::Pyramid<double, 4, 4, 3>;
  using Coord = typename Gauss::Global;
  auto a = 2.0, b = 3.0, h = 4.0;
  auto coordinate = Coordinate{
    Coord(-a, -b, 0), Coord(+a, -b, 0), Coord(+a, +b, 0), Coord(-a, +b, 0),
    Coord(0, 0, h),
    Coord(0, -b, 0), Coord(+a, 0, 0), Coord(0, +b, 0), Coord(-a, 0, 0),
    Coord(-a/2, -b/2, h/2), Coord(+a/2, -b/2, h/2),
    Coord(+a/2, +b/2, h/2), Coord(-a/2, +b/2, h/2),
  };
  auto gauss = Gauss(coordinate);
  static_assert(gauss.CellDim() == 3);
  static_assert(gauss.PhysDim() == 3);
  auto volume = (a + a) * (b + b) * h / 3;
  EXPECT_NEAR(gauss.volume(), volume, 1e-13);
  EXPECT_EQ(gauss.CountPoints(), 48);
  EXPECT_NEAR(Integrate([](Coord const&){ return 2.0; }, gauss),
      volume * 2, 1e-13);
  auto f = [](Coord const& xyz){ return xyz[0]; };
  auto g = [](Coord const& xyz){ return xyz[1]; };
  auto fg = [](Coord const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, gauss), Integrate(fg, gauss));
  EXPECT_DOUBLE_EQ(Norm(f, gauss), std::sqrt(Innerprod(f, f, gauss)));
  EXPECT_DOUBLE_EQ(Norm(g, gauss), std::sqrt(Innerprod(g, g, gauss)));
}
TEST_F(TestGaussPyramid, On14NodeQuadraticElement) {
  using Coordinate = mini::coordinate::Pyramid14<double>;
  using Gauss = mini::integrator::Pyramid<double, 4, 4, 3>;
  using Coord = typename Gauss::Global;
  auto a = 2.0, b = 3.0, h = 4.0;
  auto coordinate = Coordinate{
    Coord(-a, -b, 0), Coord(+a, -b, 0), Coord(+a, +b, 0), Coord(-a, +b, 0),
    Coord(0, 0, h),
    Coord(0, -b, 0), Coord(+a, 0, 0), Coord(0, +b, 0), Coord(-a, 0, 0),
    Coord(-a/2, -b/2, h/2), Coord(+a/2, -b/2, h/2),
    Coord(+a/2, +b/2, h/2), Coord(-a/2, +b/2, h/2),
    Coord(0, 0, 0),
  };
  auto gauss = Gauss(coordinate);
  static_assert(gauss.CellDim() == 3);
  static_assert(gauss.PhysDim() == 3);
  auto volume = (a + a) * (b + b) * h / 3;
  EXPECT_NEAR(gauss.volume(), volume, 1e-13);
  EXPECT_EQ(gauss.CountPoints(), 48);
  EXPECT_NEAR(Integrate([](Coord const&){ return 2.0; }, gauss),
      volume * 2, 1e-13);
  auto f = [](Coord const& xyz){ return xyz[0]; };
  auto g = [](Coord const& xyz){ return xyz[1]; };
  auto fg = [](Coord const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, gauss), Integrate(fg, gauss));
  EXPECT_DOUBLE_EQ(Norm(f, gauss), std::sqrt(Innerprod(f, f, gauss)));
  EXPECT_DOUBLE_EQ(Norm(g, gauss), std::sqrt(Innerprod(g, g, gauss)));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
