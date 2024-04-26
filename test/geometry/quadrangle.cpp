//  Copyright 2023 PEI Weicheng

#include <numeric>
#include <cmath>

#include "mini/geometry/face.hpp"
#include "mini/geometry/quadrangle.hpp"

#include "gtest/gtest.h"

class TestCoordinateQuadrangle : public ::testing::Test {
 protected:
};
TEST_F(TestCoordinateQuadrangle, TwoDimensionalQuadrangle4) {
  using Coordinate = mini::geometry::Quadrangle4<double, 2>;
  using Coord = typename Coordinate::Global;
  using Local = typename Coordinate::Local;
  auto face = Coordinate {
    Coord(-10, -10), Coord(+10, -10), Coord(+10, +10), Coord(-10, +10),
  };
  static_assert(face.CellDim() == 2);
  static_assert(face.PhysDim() == 2);
  EXPECT_EQ(face.CountCorners(), 4);
  EXPECT_EQ(face.CountNodes(), 4);
  EXPECT_EQ(face.LocalToGlobal(1, 1), Coord(10, 10));
  EXPECT_EQ(face.LocalToGlobal(1.5, 1.5), Coord(15, 15));
  EXPECT_EQ(face.LocalToGlobal(3, 4), Coord(30, 40));
  EXPECT_NEAR(0, (face.GlobalToLocal(30, 40) - Local(3, 4)).norm(), 1e-15);
  EXPECT_NEAR(0, (face.GlobalToLocal(40, 55) - Local(4, 5.5)).norm(), 1e-14);
  EXPECT_NEAR(0, (face.GlobalToLocal(70, 130) - Local(7, 13)).norm(), 1e-14);
  EXPECT_NEAR(0, (face.GlobalToLocal(-20, -10) - Local(-2, -1)).norm(), 1e-15);
  EXPECT_NEAR(0, (face.GlobalToLocal(-10, -20) - Local(-1, -2)).norm(), 1e-15);
  EXPECT_NEAR(0, (face.GlobalToLocal(+20, -10) - Local(+2, -1)).norm(), 1e-15);
  EXPECT_NEAR(0, (face.GlobalToLocal(-10, +20) - Local(-1, +2)).norm(), 1e-15);
  EXPECT_NEAR(0, (face.GlobalToLocal(-20, +10) - Local(-2, +1)).norm(), 1e-15);
  EXPECT_NEAR(0, (face.GlobalToLocal(+10, -20) - Local(+1, -2)).norm(), 1e-15);
  EXPECT_NEAR(0, (face.GlobalToLocal(+20, +10) - Local(+2, +1)).norm(), 1e-15);
  EXPECT_NEAR(0, (face.GlobalToLocal(+10, +20) - Local(+1, +2)).norm(), 1e-15);
  face = Coordinate {
    Coord(2.00000, 1.000000), Coord(1.94313, 0.878607),
    Coord(2.00493, 0.845382), Coord(2.06283, 0.874438),
  };
  EXPECT_ANY_THROW(face.GlobalToLocal(2.05723, 0.777978));
}
TEST_F(TestCoordinateQuadrangle, ThreeDimensionalQuadrangle4) {
  constexpr int D = 3;
  using Coordinate = mini::geometry::Quadrangle4<double, D>;
  using Coord = typename Coordinate::Global;
  using Local = typename Coordinate::Local;
  auto quadrangle = Coordinate {
    Coord(10, 0, 0), Coord(0, 10, 0), Coord(-10, 10, 10), Coord(0, 0, 10)
  };
  static_assert(quadrangle.CellDim() == 2);
  static_assert(quadrangle.PhysDim() == 3);
  EXPECT_EQ(quadrangle.CountCorners(), 4);
  EXPECT_EQ(quadrangle.CountNodes(), 4);
  EXPECT_EQ(quadrangle.center(), Coord(0, 5, 5));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(0)),
                                    quadrangle.GetGlobalCoord(0));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(1)),
                                    quadrangle.GetGlobalCoord(1));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(2)),
                                    quadrangle.GetGlobalCoord(2));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(3)),
                                    quadrangle.GetGlobalCoord(3));
  mini::geometry::Face<typename Coordinate::Real, D> &face = quadrangle;
  // test the partition-of-unity property:
  std::srand(31415926);
  auto rand = [](){ return -1 + 2.0 * std::rand() / (1.0 + RAND_MAX); };
  for (int i = 0; i < 1000; ++i) {
    auto x = rand(), y = rand();
    auto shapes = face.LocalToShapeFunctions(x, y);
    auto sum = std::accumulate(shapes.begin(), shapes.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-15);
    auto grads = face.LocalToShapeGradients(x, y);
    Local grads_sum
        = std::accumulate(grads.begin(), grads.end(), Local(0, 0));
    EXPECT_NEAR(grads_sum.norm(), 0.0, 1e-15);
    // compare gradients with O(h^2) finite difference derivatives
    int X{0}, Y{1};
    auto h = 1e-6;
    int n_node = face.CountNodes();
    auto left = face.LocalToShapeFunctions(x - h, y);
    auto right = face.LocalToShapeFunctions(x + h, y);
    for (int i_node = 0; i_node < n_node; ++i_node) {
      grads[i_node][X] -= (right[i_node] - left[i_node]) / (2 * h);
    }
    left = face.LocalToShapeFunctions(x, y - h);
    right = face.LocalToShapeFunctions(x, y + h);
    for (int i_node = 0; i_node < n_node; ++i_node) {
      grads[i_node][Y] -= (right[i_node] - left[i_node]) / (2 * h);
      EXPECT_NEAR(grads[i_node].norm(), 0.0, 1e-9);
    }
  }
  // test the Kronecker-delta and property:
  for (int i = 0, n = face.CountNodes(); i < n; ++i) {
    auto local_i = face.GetLocalCoord(i);
    auto shapes = face.LocalToShapeFunctions(local_i);
    for (int j = 0; j < n; ++j) {
      EXPECT_EQ(shapes[j], i == j);
    }
  }
  // test normal frames:
  auto normal = Coord(1, 1, 1).normalized();
  for (int i = 0; i < 1000; ++i) {
    auto x = rand(), y = rand();
    auto frame = face.LocalToNormalFrame(x, y);
    EXPECT_NEAR((frame[0] - normal).norm(), 0.0, 1e-15);
    EXPECT_NEAR(frame[0].dot(frame[1]), 0.0, 1e-15);
    EXPECT_NEAR(frame[0].dot(frame[2]), 0.0, 1e-15);
    EXPECT_NEAR(frame[1].dot(frame[2]), 0.0, 1e-15);
  }
}
TEST_F(TestCoordinateQuadrangle, ThreeDimensionalQuadrangl8) {
  constexpr int D = 3;
  using Coordinate = mini::geometry::Quadrangle8<double, D>;
  using Coord = typename Coordinate::Global;
  using Local = typename Coordinate::Local;
  auto quadrangle = Coordinate {
    Coord(10, 0, 0), Coord(0, 10, 0), Coord(-10, 10, 10), Coord(0, 0, 10),
    Coord(5, 5, 0), Coord(-5, 10, 5), Coord(-5, 5, 10), Coord(5, 0, 5)
  };
  static_assert(quadrangle.CellDim() == 2);
  static_assert(quadrangle.PhysDim() == 3);
  EXPECT_EQ(quadrangle.CountCorners(), 4);
  EXPECT_EQ(quadrangle.CountNodes(), 8);
  EXPECT_EQ(quadrangle.center(), Coord(0, 5, 5));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(0)),
                                    quadrangle.GetGlobalCoord(0));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(1)),
                                    quadrangle.GetGlobalCoord(1));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(2)),
                                    quadrangle.GetGlobalCoord(2));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(3)),
                                    quadrangle.GetGlobalCoord(3));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(4)),
                                    quadrangle.GetGlobalCoord(4));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(5)),
                                    quadrangle.GetGlobalCoord(5));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(6)),
                                    quadrangle.GetGlobalCoord(6));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(7)),
                                    quadrangle.GetGlobalCoord(7));
  mini::geometry::Face<typename Coordinate::Real, D> &face = quadrangle;
  // test the partition-of-unity property:
  std::srand(31415926);
  auto rand = [](){ return -1 + 2.0 * std::rand() / (1.0 + RAND_MAX); };
  for (int i = 0; i < 1000; ++i) {
    auto x = rand(), y = rand();
    auto shapes = face.LocalToShapeFunctions(x, y);
    auto sum = std::accumulate(shapes.begin(), shapes.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-14);
    auto grads = face.LocalToShapeGradients(x, y);
    Local grads_sum
        = std::accumulate(grads.begin(), grads.end(), Local(0, 0));
    EXPECT_NEAR(grads_sum.norm(), 0.0, 1e-15);
    // compare gradients with O(h^2) finite difference derivatives
    int X{0}, Y{1};
    auto h = 1e-6;
    int n_node = face.CountNodes();
    auto left = face.LocalToShapeFunctions(x - h, y);
    auto right = face.LocalToShapeFunctions(x + h, y);
    for (int i_node = 0; i_node < n_node; ++i_node) {
      grads[i_node][X] -= (right[i_node] - left[i_node]) / (2 * h);
    }
    left = face.LocalToShapeFunctions(x, y - h);
    right = face.LocalToShapeFunctions(x, y + h);
    for (int i_node = 0; i_node < n_node; ++i_node) {
      grads[i_node][Y] -= (right[i_node] - left[i_node]) / (2 * h);
      EXPECT_NEAR(grads[i_node].norm(), 0.0, 1e-9);
    }
  }
  // test the Kronecker-delta and property:
  for (int i = 0, n = face.CountNodes(); i < n; ++i) {
    auto local_i = face.GetLocalCoord(i);
    auto shapes = face.LocalToShapeFunctions(local_i);
    for (int j = 0; j < n; ++j) {
      EXPECT_EQ(shapes[j], i == j);
    }
  }
  // test normal frames:
  auto normal = Coord(1, 1, 1).normalized();
  for (int i = 0; i < 1000; ++i) {
    auto x = rand(), y = rand();
    auto frame = face.LocalToNormalFrame(x, y);
    EXPECT_NEAR((frame[0] - normal).norm(), 0.0, 1e-15);
    EXPECT_NEAR(frame[0].dot(frame[1]), 0.0, 1e-15);
    EXPECT_NEAR(frame[0].dot(frame[2]), 0.0, 1e-15);
    EXPECT_NEAR(frame[1].dot(frame[2]), 0.0, 1e-15);
  }
}
TEST_F(TestCoordinateQuadrangle, ThreeDimensionalQuadrangle9) {
  constexpr int D = 3;
  using Coordinate = mini::geometry::Quadrangle9<double, D>;
  using Coord = typename Coordinate::Global;
  using Local = typename Coordinate::Local;
  auto quadrangle = Coordinate {
    Coord(10, 0, 0), Coord(0, 10, 0), Coord(-10, 10, 10), Coord(0, 0, 10),
    Coord(5, 5, 0), Coord(-5, 10, 5), Coord(-5, 5, 10), Coord(5, 0, 5),
    Coord(0, 5, 5)
  };
  static_assert(quadrangle.CellDim() == 2);
  static_assert(quadrangle.PhysDim() == 3);
  EXPECT_EQ(quadrangle.CountCorners(), 4);
  EXPECT_EQ(quadrangle.CountNodes(), 9);
  EXPECT_EQ(quadrangle.center(), Coord(0, 5, 5));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(0)),
                                    quadrangle.GetGlobalCoord(0));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(1)),
                                    quadrangle.GetGlobalCoord(1));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(2)),
                                    quadrangle.GetGlobalCoord(2));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(3)),
                                    quadrangle.GetGlobalCoord(3));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(4)),
                                    quadrangle.GetGlobalCoord(4));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(5)),
                                    quadrangle.GetGlobalCoord(5));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(6)),
                                    quadrangle.GetGlobalCoord(6));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocalCoord(7)),
                                    quadrangle.GetGlobalCoord(7));
  mini::geometry::Face<typename Coordinate::Real, D> &face = quadrangle;
  // test the partition-of-unity property:
  std::srand(31415926);
  auto rand = [](){ return -1 + 2.0 * std::rand() / (1.0 + RAND_MAX); };
  for (int i = 0; i < 1000; ++i) {
    auto x = rand(), y = rand();
    auto shapes = face.LocalToShapeFunctions(x, y);
    auto sum = std::accumulate(shapes.begin(), shapes.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-14);
    auto grads = face.LocalToShapeGradients(x, y);
    Local grads_sum
        = std::accumulate(grads.begin(), grads.end(), Local(0, 0));
    EXPECT_NEAR(grads_sum.norm(), 0.0, 1e-15);
  }
  // test the Kronecker-delta and property:
  for (int i = 0, n = face.CountNodes(); i < n; ++i) {
    auto local_i = face.GetLocalCoord(i);
    auto shapes = face.LocalToShapeFunctions(local_i);
    for (int j = 0; j < n; ++j) {
      EXPECT_EQ(shapes[j], i == j);
    }
  }
  // test normal frames:
  auto normal = Coord(1, 1, 1).normalized();
  for (int i = 0; i < 1000; ++i) {
    auto x = rand(), y = rand();
    auto frame = face.LocalToNormalFrame(x, y);
    EXPECT_NEAR((frame[0] - normal).norm(), 0.0, 1e-15);
    EXPECT_NEAR(frame[0].dot(frame[1]), 0.0, 1e-15);
    EXPECT_NEAR(frame[0].dot(frame[2]), 0.0, 1e-15);
    EXPECT_NEAR(frame[1].dot(frame[2]), 0.0, 1e-15);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
