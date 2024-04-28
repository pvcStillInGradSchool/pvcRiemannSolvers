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
  auto integrator = Integrator(coordinate);
  static_assert(integrator.CellDim() == 3);
  static_assert(integrator.PhysDim() == 3);
  EXPECT_NEAR(integrator.volume(), 4.0, 1e-14);
  EXPECT_EQ(integrator.CountPoints(), 64);
  auto local_weight_sum = 0.0;
  for (int i = 0; i < integrator.CountPoints(); ++i) {
    local_weight_sum += integrator.GetLocalWeight(i);
  }
  EXPECT_NEAR(Quadrature([](Coord const&){ return 2.0; }, integrator),
      local_weight_sum * 2, 1e-15);
  EXPECT_NEAR(Integrate([](Coord const&){ return 2.0; }, integrator),
      integrator.volume() * 2, 1e-15);
  auto f = [](Coord const& xyz){ return xyz[0]; };
  auto g = [](Coord const& xyz){ return xyz[1]; };
  auto h = [](Coord const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, integrator), Integrate(h, integrator));
  EXPECT_DOUBLE_EQ(Norm(f, integrator), std::sqrt(Innerprod(f, f, integrator)));
  EXPECT_DOUBLE_EQ(Norm(g, integrator), std::sqrt(Innerprod(g, g, integrator)));
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
  auto integrator = Integrator(coordinate);
  static_assert(integrator.CellDim() == 3);
  static_assert(integrator.PhysDim() == 3);
  EXPECT_NEAR(integrator.volume(), 4.0, 1e-14);
  EXPECT_EQ(integrator.CountPoints(), 64);
  auto local_weight_sum = 0.0;
  for (int i = 0; i < integrator.CountPoints(); ++i) {
    local_weight_sum += integrator.GetLocalWeight(i);
  }
  EXPECT_NEAR(Quadrature([](Coord const&){ return 2.0; }, integrator),
      local_weight_sum * 2, 1e-15);
  EXPECT_NEAR(Integrate([](Coord const&){ return 2.0; }, integrator),
      integrator.volume() * 2, 1e-15);
  auto f = [](Coord const& xyz){ return xyz[0]; };
  auto g = [](Coord const& xyz){ return xyz[1]; };
  auto h = [](Coord const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, integrator), Integrate(h, integrator));
  EXPECT_DOUBLE_EQ(Norm(f, integrator), std::sqrt(Innerprod(f, f, integrator)));
  EXPECT_DOUBLE_EQ(Norm(g, integrator), std::sqrt(Innerprod(g, g, integrator)));
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
  auto integrator = Integrator(coordinate);
  static_assert(integrator.CellDim() == 3);
  static_assert(integrator.PhysDim() == 3);
  EXPECT_NEAR(integrator.volume(), 4.0, 1e-14);
  EXPECT_EQ(integrator.CountPoints(), 64);
  auto local_weight_sum = 0.0;
  for (int i = 0; i < integrator.CountPoints(); ++i) {
    local_weight_sum += integrator.GetLocalWeight(i);
  }
  EXPECT_NEAR(Quadrature([](Coord const&){ return 2.0; }, integrator),
      local_weight_sum * 2, 1e-15);
  EXPECT_NEAR(Integrate([](Coord const&){ return 2.0; }, integrator),
      integrator.volume() * 2, 1e-15);
  auto f = [](Coord const& xyz){ return xyz[0]; };
  auto g = [](Coord const& xyz){ return xyz[1]; };
  auto h = [](Coord const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, integrator), Integrate(h, integrator));
  EXPECT_DOUBLE_EQ(Norm(f, integrator), std::sqrt(Innerprod(f, f, integrator)));
  EXPECT_DOUBLE_EQ(Norm(g, integrator), std::sqrt(Innerprod(g, g, integrator)));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
