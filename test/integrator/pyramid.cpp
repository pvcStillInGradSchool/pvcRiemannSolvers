//  Copyright 2023 PEI Weicheng

#include <cmath>

#include "mini/integrator/function.hpp"
#include "mini/integrator/legendre.hpp"
#include "mini/integrator/pyramid.hpp"
#include "mini/coordinate/pyramid.hpp"

#include "gtest/gtest.h"

class TestIntegratorPyramid : public ::testing::Test {
 protected:
  using Legendre4 = mini::integrator::Legendre<double, 4>;
};
TEST_F(TestIntegratorPyramid, OnLinearElement) {
  using Coordinate = mini::coordinate::Pyramid5<double>;
  using Integrator = mini::integrator::Pyramid<Legendre4, Legendre4, 3>;
  using Coord = typename Integrator::Global;
  auto a = 2.0, b = 3.0, h = 4.0;
  auto coordinate = Coordinate{
    Coord(-a, -b, 0), Coord(+a, -b, 0),
    Coord(+a, +b, 0), Coord(-a, +b, 0),
    Coord(0, 0, h)
  };
  auto integrator = Integrator(coordinate);
  static_assert(integrator.CellDim() == 3);
  static_assert(integrator.PhysDim() == 3);
  auto volume = (a + a) * (b + b) * h / 3;
  EXPECT_NEAR(integrator.volume(), volume, 1e-13);
  EXPECT_EQ(integrator.CountPoints(), 48);
  EXPECT_NEAR(Integrate([](Coord const&){ return 2.0; }, integrator),
      volume * 2, 1e-13);
  auto f = [](Coord const& xyz){ return xyz[0]; };
  auto g = [](Coord const& xyz){ return xyz[1]; };
  auto fg = [](Coord const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, integrator), Integrate(fg, integrator));
  EXPECT_DOUBLE_EQ(Norm(f, integrator), std::sqrt(Innerprod(f, f, integrator)));
  EXPECT_DOUBLE_EQ(Norm(g, integrator), std::sqrt(Innerprod(g, g, integrator)));
}
TEST_F(TestIntegratorPyramid, OnQuadraticElement) {
  using Coordinate = mini::coordinate::Pyramid13<double>;
  using Integrator = mini::integrator::Pyramid<Legendre4, Legendre4, 3>;
  using Coord = typename Integrator::Global;
  auto a = 2.0, b = 3.0, h = 4.0;
  auto coordinate = Coordinate{
    Coord(-a, -b, 0), Coord(+a, -b, 0), Coord(+a, +b, 0), Coord(-a, +b, 0),
    Coord(0, 0, h),
    Coord(0, -b, 0), Coord(+a, 0, 0), Coord(0, +b, 0), Coord(-a, 0, 0),
    Coord(-a/2, -b/2, h/2), Coord(+a/2, -b/2, h/2),
    Coord(+a/2, +b/2, h/2), Coord(-a/2, +b/2, h/2),
  };
  auto integrator = Integrator(coordinate);
  static_assert(integrator.CellDim() == 3);
  static_assert(integrator.PhysDim() == 3);
  auto volume = (a + a) * (b + b) * h / 3;
  EXPECT_NEAR(integrator.volume(), volume, 1e-13);
  EXPECT_EQ(integrator.CountPoints(), 48);
  EXPECT_NEAR(Integrate([](Coord const&){ return 2.0; }, integrator),
      volume * 2, 1e-13);
  auto f = [](Coord const& xyz){ return xyz[0]; };
  auto g = [](Coord const& xyz){ return xyz[1]; };
  auto fg = [](Coord const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, integrator), Integrate(fg, integrator));
  EXPECT_DOUBLE_EQ(Norm(f, integrator), std::sqrt(Innerprod(f, f, integrator)));
  EXPECT_DOUBLE_EQ(Norm(g, integrator), std::sqrt(Innerprod(g, g, integrator)));
}
TEST_F(TestIntegratorPyramid, On14NodeQuadraticElement) {
  using Coordinate = mini::coordinate::Pyramid14<double>;
  using Integrator = mini::integrator::Pyramid<Legendre4, Legendre4, 3>;
  using Coord = typename Integrator::Global;
  auto a = 2.0, b = 3.0, h = 4.0;
  auto coordinate = Coordinate{
    Coord(-a, -b, 0), Coord(+a, -b, 0), Coord(+a, +b, 0), Coord(-a, +b, 0),
    Coord(0, 0, h),
    Coord(0, -b, 0), Coord(+a, 0, 0), Coord(0, +b, 0), Coord(-a, 0, 0),
    Coord(-a/2, -b/2, h/2), Coord(+a/2, -b/2, h/2),
    Coord(+a/2, +b/2, h/2), Coord(-a/2, +b/2, h/2),
    Coord(0, 0, 0),
  };
  auto integrator = Integrator(coordinate);
  static_assert(integrator.CellDim() == 3);
  static_assert(integrator.PhysDim() == 3);
  auto volume = (a + a) * (b + b) * h / 3;
  EXPECT_NEAR(integrator.volume(), volume, 1e-13);
  EXPECT_EQ(integrator.CountPoints(), 48);
  EXPECT_NEAR(Integrate([](Coord const&){ return 2.0; }, integrator),
      volume * 2, 1e-13);
  auto f = [](Coord const& xyz){ return xyz[0]; };
  auto g = [](Coord const& xyz){ return xyz[1]; };
  auto fg = [](Coord const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, integrator), Integrate(fg, integrator));
  EXPECT_DOUBLE_EQ(Norm(f, integrator), std::sqrt(Innerprod(f, f, integrator)));
  EXPECT_DOUBLE_EQ(Norm(g, integrator), std::sqrt(Innerprod(g, g, integrator)));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
