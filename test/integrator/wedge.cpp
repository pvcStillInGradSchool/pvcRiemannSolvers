//  Copyright 2023 PEI Weicheng

#include <cmath>

#include "mini/integrator/function.hpp"
#include "mini/integrator/wedge.hpp"
#include "mini/coordinate/wedge.hpp"

#include "gtest/gtest.h"

class TestIntegratorWedge : public ::testing::Test {
};
TEST_F(TestIntegratorWedge, OnLinearElement) {
  using Integrator = mini::integrator::Wedge<double, 16, 4>;
  using Coordinate = mini::coordinate::Wedge6<double>;
  using Coord = typename Coordinate::Global;
  auto coordinate = Coordinate {
    Coord(-1, -1, -1), Coord(+1, -1, -1), Coord(0, +1, -1),
    Coord(-1, -1, +1), Coord(+1, -1, +1), Coord(0, +1, +1)
  };
  auto gauss = Integrator(coordinate);
  static_assert(gauss.CellDim() == 3);
  static_assert(gauss.PhysDim() == 3);
  EXPECT_NEAR(gauss.volume(), 4.0, 1e-14);
  EXPECT_EQ(gauss.CountPoints(), 64);
  auto local_weight_sum = 0.0;
  for (int i = 0; i < gauss.CountPoints(); ++i) {
    local_weight_sum += gauss.GetLocalWeight(i);
  }
  EXPECT_NEAR(Quadrature([](Coord const&){ return 2.0; }, gauss),
      local_weight_sum * 2, 1e-15);
  EXPECT_NEAR(Integrate([](Coord const&){ return 2.0; }, gauss),
      gauss.volume() * 2, 1e-15);
  auto f = [](Coord const& xyz){ return xyz[0]; };
  auto g = [](Coord const& xyz){ return xyz[1]; };
  auto h = [](Coord const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, gauss), Integrate(h, gauss));
  EXPECT_DOUBLE_EQ(Norm(f, gauss), std::sqrt(Innerprod(f, f, gauss)));
  EXPECT_DOUBLE_EQ(Norm(g, gauss), std::sqrt(Innerprod(g, g, gauss)));
}
TEST_F(TestIntegratorWedge, OnQuadraticElement) {
  using Integrator = mini::integrator::Wedge<double, 16, 4>;
  using Coordinate = mini::coordinate::Wedge15<double>;
  using Coord = typename Coordinate::Global;
  auto coordinate = Coordinate {
    Coord(-1, -1, -1), Coord(+1, -1, -1), Coord(0, +1, -1),
    Coord(-1, -1, +1), Coord(+1, -1, +1), Coord(0, +1, +1),
    Coord(0, -1, -1), Coord(0.5, 0, -1), Coord(-0.5, 0, -1),
    Coord(-1, -1, 0), Coord(+1, -1, 0), Coord(0, +1, 0),
    Coord(0, -1, +1), Coord(0.5, 0, +1), Coord(-0.5, 0, +1),
  };
  auto gauss = Integrator(coordinate);
  static_assert(gauss.CellDim() == 3);
  static_assert(gauss.PhysDim() == 3);
  EXPECT_NEAR(gauss.volume(), 4.0, 1e-14);
  EXPECT_EQ(gauss.CountPoints(), 64);
  auto local_weight_sum = 0.0;
  for (int i = 0; i < gauss.CountPoints(); ++i) {
    local_weight_sum += gauss.GetLocalWeight(i);
  }
  EXPECT_NEAR(Quadrature([](Coord const&){ return 2.0; }, gauss),
      local_weight_sum * 2, 1e-15);
  EXPECT_NEAR(Integrate([](Coord const&){ return 2.0; }, gauss),
      gauss.volume() * 2, 1e-15);
  auto f = [](Coord const& xyz){ return xyz[0]; };
  auto g = [](Coord const& xyz){ return xyz[1]; };
  auto h = [](Coord const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, gauss), Integrate(h, gauss));
  EXPECT_DOUBLE_EQ(Norm(f, gauss), std::sqrt(Innerprod(f, f, gauss)));
  EXPECT_DOUBLE_EQ(Norm(g, gauss), std::sqrt(Innerprod(g, g, gauss)));
}
TEST_F(TestIntegratorWedge, On18NodeQuadraticElement) {
  using Integrator = mini::integrator::Wedge<double, 16, 4>;
  using Coordinate = mini::coordinate::Wedge18<double>;
  using Coord = typename Coordinate::Global;
  auto coordinate = Coordinate {
    Coord(-1, -1, -1), Coord(+1, -1, -1), Coord(0, +1, -1),
    Coord(-1, -1, +1), Coord(+1, -1, +1), Coord(0, +1, +1),
    Coord(0, -1, -1), Coord(0.5, 0, -1), Coord(-0.5, 0, -1),
    Coord(-1, -1, 0), Coord(+1, -1, 0), Coord(0, +1, 0),
    Coord(0, -1, +1), Coord(0.5, 0, +1), Coord(-0.5, 0, +1),
    Coord(0, -1, 0), Coord(0.5, 0, 0), Coord(-0.5, 0, 0),
  };
  auto gauss = Integrator(coordinate);
  static_assert(gauss.CellDim() == 3);
  static_assert(gauss.PhysDim() == 3);
  EXPECT_NEAR(gauss.volume(), 4.0, 1e-14);
  EXPECT_EQ(gauss.CountPoints(), 64);
  auto local_weight_sum = 0.0;
  for (int i = 0; i < gauss.CountPoints(); ++i) {
    local_weight_sum += gauss.GetLocalWeight(i);
  }
  EXPECT_NEAR(Quadrature([](Coord const&){ return 2.0; }, gauss),
      local_weight_sum * 2, 1e-15);
  EXPECT_NEAR(Integrate([](Coord const&){ return 2.0; }, gauss),
      gauss.volume() * 2, 1e-15);
  auto f = [](Coord const& xyz){ return xyz[0]; };
  auto g = [](Coord const& xyz){ return xyz[1]; };
  auto h = [](Coord const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, gauss), Integrate(h, gauss));
  EXPECT_DOUBLE_EQ(Norm(f, gauss), std::sqrt(Innerprod(f, f, gauss)));
  EXPECT_DOUBLE_EQ(Norm(g, gauss), std::sqrt(Innerprod(g, g, gauss)));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
