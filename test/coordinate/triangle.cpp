//  Copyright 2023 PEI Weicheng

#include <numeric>

#include <cmath>

#include "mini/coordinate/face.hpp"
#include "mini/coordinate/triangle.hpp"

#include "gtest/gtest.h"

class TestCoordinateTriangle3 : public ::testing::Test {
 protected:
};
TEST_F(TestCoordinateTriangle3, ThreeDimensional) {
  constexpr int D = 3;
  using Coordinate = mini::coordinate::Triangle3<double, D>;
  using Coord = typename Coordinate::Global;
  using Local = typename Coordinate::Local;
  auto quadrangle = Coordinate {
    Coord(9, 0, 0), Coord(0, 9, 0), Coord(0, 0, 9)
  };
  static_assert(quadrangle.CellDim() == 2);
  static_assert(quadrangle.PhysDim() == 3);
  EXPECT_EQ(quadrangle.CountCorners(), 3);
  EXPECT_EQ(quadrangle.CountNodes(), 3);
  EXPECT_NEAR((quadrangle.center() - Coord(3, 3, 3)).norm(), 0.0, 1e-15);
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocal(0)),
                                    quadrangle.GetGlobal(0));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocal(1)),
                                    quadrangle.GetGlobal(1));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocal(2)),
                                    quadrangle.GetGlobal(2));
  mini::coordinate::Face<typename Coordinate::Real, D> &face = quadrangle;
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
    auto local_i = face.GetLocal(i);
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

class TestCoordinateTriangle6 : public ::testing::Test {
 protected:
};
TEST_F(TestCoordinateTriangle6, ThreeDimensional) {
  constexpr int D = 3;
  using Coordinate = mini::coordinate::Triangle6<double, D>;
  using Coord = typename Coordinate::Global;
  using Local = typename Coordinate::Local;
  auto quadrangle = Coordinate {
    Coord(9, 0, 0), Coord(0, 9, 0), Coord(0, 0, 9),
    Coord(4.5, 4.5, 0), Coord(0, 4.5, 4.5), Coord(4.5, 0, 4.5)
  };
  static_assert(quadrangle.CellDim() == 2);
  static_assert(quadrangle.PhysDim() == 3);
  EXPECT_EQ(quadrangle.CountCorners(), 3);
  EXPECT_EQ(quadrangle.CountNodes(), 6);
  EXPECT_NEAR((quadrangle.center() - Coord(3, 3, 3)).norm(), 0.0, 1e-14);
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocal(0)),
                                    quadrangle.GetGlobal(0));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocal(1)),
                                    quadrangle.GetGlobal(1));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocal(2)),
                                    quadrangle.GetGlobal(2));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocal(3)),
                                    quadrangle.GetGlobal(3));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocal(4)),
                                    quadrangle.GetGlobal(4));
  EXPECT_EQ(quadrangle.LocalToGlobal(quadrangle.GetLocal(5)),
                                    quadrangle.GetGlobal(5));
  mini::coordinate::Face<typename Coordinate::Real, D> &face = quadrangle;
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
      EXPECT_NEAR(grads[i_node].norm(), 0.0, 1e-8);
    }
  }
  // test the Kronecker-delta and property:
  for (int i = 0, n = face.CountNodes(); i < n; ++i) {
    auto local_i = face.GetLocal(i);
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
