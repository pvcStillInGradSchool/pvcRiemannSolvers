//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/gauss/function.hpp"
#include "mini/gauss/legendre.hpp"
#include "mini/gauss/hexahedron.hpp"
#include "mini/geometry/hexahedron.hpp"

#include "gtest/gtest.h"

class TestGaussHexahedron : public ::testing::Test {
 protected:
  using Gx = mini::gauss::Legendre<double, 4>;
  using Gy = mini::gauss::Legendre<double, 4>;
  using Gz = mini::gauss::Legendre<double, 4>;
  using Gauss = mini::gauss::Hexahedron<Gx, Gy, Gz>;
  using Coord = typename Gauss::Global;
};
TEST_F(TestGaussHexahedron, OnLinearElement) {
  using Coordinate = mini::geometry::Hexahedron8<double>;
  auto lagrange = Coordinate {
    Coord(-1, -1, -1), Coord(+1, -1, -1),
    Coord(+1, +1, -1), Coord(-1, +1, -1),
    Coord(-1, -1, +1), Coord(+1, -1, +1),
    Coord(+1, +1, +1), Coord(-1, +1, +1)
  };
  auto hexa = Gauss(lagrange);
  static_assert(hexa.CellDim() == 3);
  static_assert(hexa.PhysDim() == 3);
  EXPECT_NEAR(hexa.volume(), 8.0, 1e-14);
  EXPECT_EQ(hexa.CountPoints(), 64);
  auto p0 = hexa.GetLocalCoord(0);
  EXPECT_EQ(p0[0], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  EXPECT_EQ(p0[1], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  EXPECT_EQ(p0[2], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  auto w1d = (18 - std::sqrt(30)) / 36.0;
  EXPECT_EQ(hexa.GetLocalWeight(0), w1d * w1d * w1d);
  lagrange = Coordinate {
    Coord(-10, -10, -10), Coord(+10, -10, -10),
    Coord(+10, +10, -10), Coord(-10, +10, -10),
    Coord(-10, -10, +10), Coord(+10, -10, +10),
    Coord(+10, +10, +10), Coord(-10, +10, +10)
  };
  hexa = Gauss(lagrange);
  EXPECT_DOUBLE_EQ(Quadrature([](Coord const&){ return 2.0; }, hexa), 16.0);
  EXPECT_NEAR(Integrate([](Coord const&){ return 2.0; }, hexa), 16000, 1e-10);
  auto f = [](Coord const& xyz){ return xyz[0]; };
  auto g = [](Coord const& xyz){ return xyz[1]; };
  auto h = [](Coord const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, hexa), Integrate(h, hexa));
  EXPECT_DOUBLE_EQ(Norm(f, hexa), std::sqrt(Innerprod(f, f, hexa)));
  EXPECT_DOUBLE_EQ(Norm(g, hexa), std::sqrt(Innerprod(g, g, hexa)));
}
TEST_F(TestGaussHexahedron, OnQuadraticElement) {
  using Coordinate = mini::geometry::Hexahedron20<double>;
  auto lagrange = Coordinate {
    Coord(-1, -1, -1), Coord(+1, -1, -1), Coord(+1, +1, -1), Coord(-1, +1, -1),
    Coord(-1, -1, +1), Coord(+1, -1, +1), Coord(+1, +1, +1), Coord(-1, +1, +1),
    Coord(0, -1, -1), Coord(+1, 0, -1), Coord(0, +1, -1), Coord(-1, 0, -1),
    Coord(-1, -1, 0), Coord(+1, -1, 0), Coord(+1, +1, 0), Coord(-1, +1, 0),
    Coord(0, -1, +1), Coord(+1, 0, +1), Coord(0, +1, +1), Coord(-1, 0, +1),
  };
  auto hexa = Gauss(lagrange);
  static_assert(hexa.CellDim() == 3);
  static_assert(hexa.PhysDim() == 3);
  EXPECT_NEAR(hexa.volume(), 8.0, 1e-14);
  EXPECT_EQ(hexa.CountPoints(), 64);
  auto p0 = hexa.GetLocalCoord(0);
  EXPECT_EQ(p0[0], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  EXPECT_EQ(p0[1], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  EXPECT_EQ(p0[2], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  auto w1d = (18 - std::sqrt(30)) / 36.0;
  EXPECT_EQ(hexa.GetLocalWeight(0), w1d * w1d * w1d);
  lagrange = Coordinate {
    // corner nodes on the bottom face
    Coord(-10, -10, -10), Coord(+10, -10, -10),
    Coord(+10, +10, -10), Coord(-10, +10, -10),
    // corner nodes on the top face
    Coord(-10, -10, +10), Coord(+10, -10, +10),
    Coord(+10, +10, +10), Coord(-10, +10, +10),
    // mid-edge nodes on the bottom face
    Coord(0, -10, -10), Coord(+10, 0, -10),
    Coord(0, +10, -10), Coord(-10, 0, -10),
    // mid-edge nodes on vertical edges
    Coord(-10, -10, 0), Coord(+10, -10, 0),
    Coord(+10, +10, 0), Coord(-10, +10, 0),
    // mid-edge nodes on the top face
    Coord(0, -10, +10), Coord(+10, 0, +10),
    Coord(0, +10, +10), Coord(-10, 0, +10),
  };
  hexa = Gauss(lagrange);
  EXPECT_DOUBLE_EQ(Quadrature([](Coord const&){ return 2.0; }, hexa), 16.0);
  EXPECT_NEAR(Integrate([](Coord const&){ return 2.0; }, hexa), 16000, 1e-10);
  auto f = [](Coord const& xyz){ return xyz[0]; };
  auto g = [](Coord const& xyz){ return xyz[1]; };
  auto h = [](Coord const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, hexa), Integrate(h, hexa));
  EXPECT_DOUBLE_EQ(Norm(f, hexa), std::sqrt(Innerprod(f, f, hexa)));
  EXPECT_DOUBLE_EQ(Norm(g, hexa), std::sqrt(Innerprod(g, g, hexa)));
}
TEST_F(TestGaussHexahedron, On27NodeQuadraticElement) {
  using Coordinate = mini::geometry::Hexahedron27<double>;
  auto lagrange = Coordinate {
    Coord(-1, -1, -1), Coord(+1, -1, -1), Coord(+1, +1, -1), Coord(-1, +1, -1),
    Coord(-1, -1, +1), Coord(+1, -1, +1), Coord(+1, +1, +1), Coord(-1, +1, +1),
    Coord(0, -1, -1), Coord(+1, 0, -1), Coord(0, +1, -1), Coord(-1, 0, -1),
    Coord(-1, -1, 0), Coord(+1, -1, 0), Coord(+1, +1, 0), Coord(-1, +1, 0),
    Coord(0, -1, +1), Coord(+1, 0, +1), Coord(0, +1, +1), Coord(-1, 0, +1),
    Coord(0, 0, -1),
    Coord(0, -1, 0), Coord(+1, 0, 0), Coord(0, +1, 0), Coord(-1, 0, 0),
    Coord(0, 0, +1), Coord(0, 0, 0),
  };
  auto hexa = Gauss(lagrange);
  static_assert(hexa.CellDim() == 3);
  static_assert(hexa.PhysDim() == 3);
  EXPECT_NEAR(hexa.volume(), 8.0, 1e-14);
  EXPECT_EQ(hexa.CountPoints(), 64);
  auto p0 = hexa.GetLocalCoord(0);
  EXPECT_EQ(p0[0], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  EXPECT_EQ(p0[1], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  EXPECT_EQ(p0[2], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  auto w1d = (18 - std::sqrt(30)) / 36.0;
  EXPECT_EQ(hexa.GetLocalWeight(0), w1d * w1d * w1d);
  lagrange = Coordinate {
    // corner nodes on the bottom face
    Coord(-10, -10, -10), Coord(+10, -10, -10),
    Coord(+10, +10, -10), Coord(-10, +10, -10),
    // corner nodes on the top face
    Coord(-10, -10, +10), Coord(+10, -10, +10),
    Coord(+10, +10, +10), Coord(-10, +10, +10),
    // mid-edge nodes on the bottom face
    Coord(0, -10, -10), Coord(+10, 0, -10),
    Coord(0, +10, -10), Coord(-10, 0, -10),
    // mid-edge nodes on vertical edges
    Coord(-10, -10, 0), Coord(+10, -10, 0),
    Coord(+10, +10, 0), Coord(-10, +10, 0),
    // mid-edge nodes on the top face
    Coord(0, -10, +10), Coord(+10, 0, +10),
    Coord(0, +10, +10), Coord(-10, 0, +10),
    // mid-face nodes
    Coord(0, 0, -10),
    Coord(0, -10, 0), Coord(+10, 0, 0), Coord(0, +10, 0), Coord(-10, 0, 0),
    Coord(0, 0, +10),
    // center
    Coord(0, 0, 0),
  };
  hexa = Gauss(lagrange);
  EXPECT_DOUBLE_EQ(Quadrature([](Coord const&){ return 2.0; }, hexa), 16.0);
  EXPECT_NEAR(Integrate([](Coord const&){ return 2.0; }, hexa), 16000, 1e-10);
  auto f = [](Coord const& xyz){ return xyz[0]; };
  auto g = [](Coord const& xyz){ return xyz[1]; };
  auto h = [](Coord const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, hexa), Integrate(h, hexa));
  EXPECT_DOUBLE_EQ(Norm(f, hexa), std::sqrt(Innerprod(f, f, hexa)));
  EXPECT_DOUBLE_EQ(Norm(g, hexa), std::sqrt(Innerprod(g, g, hexa)));
}
TEST_F(TestGaussHexahedron, On26NodeQuadraticElement) {
  using Coordinate = mini::geometry::Hexahedron26<double>;
  auto lagrange = Coordinate {
    Coord(-1, -1, -1), Coord(+1, -1, -1), Coord(+1, +1, -1), Coord(-1, +1, -1),
    Coord(-1, -1, +1), Coord(+1, -1, +1), Coord(+1, +1, +1), Coord(-1, +1, +1),
    Coord(0, -1, -1), Coord(+1, 0, -1), Coord(0, +1, -1), Coord(-1, 0, -1),
    Coord(-1, -1, 0), Coord(+1, -1, 0), Coord(+1, +1, 0), Coord(-1, +1, 0),
    Coord(0, -1, +1), Coord(+1, 0, +1), Coord(0, +1, +1), Coord(-1, 0, +1),
    Coord(0, 0, -1),
    Coord(0, -1, 0), Coord(+1, 0, 0), Coord(0, +1, 0), Coord(-1, 0, 0),
    Coord(0, 0, +1),
  };
  auto hexa = Gauss(lagrange);
  static_assert(hexa.CellDim() == 3);
  static_assert(hexa.PhysDim() == 3);
  EXPECT_NEAR(hexa.volume(), 8.0, 1e-14);
  EXPECT_EQ(hexa.CountPoints(), 64);
  auto p0 = hexa.GetLocalCoord(0);
  EXPECT_EQ(p0[0], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  EXPECT_EQ(p0[1], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  EXPECT_EQ(p0[2], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  auto w1d = (18 - std::sqrt(30)) / 36.0;
  EXPECT_EQ(hexa.GetLocalWeight(0), w1d * w1d * w1d);
  lagrange = Coordinate {
    // corner nodes on the bottom face
    Coord(-10, -10, -10), Coord(+10, -10, -10),
    Coord(+10, +10, -10), Coord(-10, +10, -10),
    // corner nodes on the top face
    Coord(-10, -10, +10), Coord(+10, -10, +10),
    Coord(+10, +10, +10), Coord(-10, +10, +10),
    // mid-edge nodes on the bottom face
    Coord(0, -10, -10), Coord(+10, 0, -10),
    Coord(0, +10, -10), Coord(-10, 0, -10),
    // mid-edge nodes on vertical edges
    Coord(-10, -10, 0), Coord(+10, -10, 0),
    Coord(+10, +10, 0), Coord(-10, +10, 0),
    // mid-edge nodes on the top face
    Coord(0, -10, +10), Coord(+10, 0, +10),
    Coord(0, +10, +10), Coord(-10, 0, +10),
    // mid-face nodes
    Coord(0, 0, -10),
    Coord(0, -10, 0), Coord(+10, 0, 0), Coord(0, +10, 0), Coord(-10, 0, 0),
    Coord(0, 0, +10),
  };
  hexa = Gauss(lagrange);
  EXPECT_DOUBLE_EQ(Quadrature([](Coord const&){ return 2.0; }, hexa), 16.0);
  EXPECT_NEAR(Integrate([](Coord const&){ return 2.0; }, hexa), 16000, 1e-10);
  auto f = [](Coord const& xyz){ return xyz[0]; };
  auto g = [](Coord const& xyz){ return xyz[1]; };
  auto h = [](Coord const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, hexa), Integrate(h, hexa));
  EXPECT_DOUBLE_EQ(Norm(f, hexa), std::sqrt(Innerprod(f, f, hexa)));
  EXPECT_DOUBLE_EQ(Norm(g, hexa), std::sqrt(Innerprod(g, g, hexa)));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
