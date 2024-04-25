//  Copyright 2023 PEI Weicheng

#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdlib>

#include "mini/geometry/cell.hpp"
#include "mini/geometry/pyramid.hpp"
#include "mini/geometry/triangle.hpp"
#include "mini/geometry/quadrangle.hpp"

#include "gtest/gtest.h"

class TestLagrangePyramid5 : public ::testing::Test {
 protected:
  using Lagrange = mini::geometry::Pyramid5<double>;
  using Coord = typename Lagrange::Global;
};
TEST_F(TestLagrangePyramid5, CoordinateMap) {
  auto a = 2.0, b = 3.0, h = 4.0;
  auto lagrange = Lagrange{
    Coord(-a, -b, 0), Coord(+a, -b, 0),
    Coord(+a, +b, 0), Coord(-a, +b, 0),
    Coord(0, 0, h)
  };
  static_assert(lagrange.CellDim() == 3);
  static_assert(lagrange.PhysDim() == 3);
  EXPECT_EQ(lagrange.CountCorners(), 5);
  EXPECT_EQ(lagrange.CountNodes(), 5);
  EXPECT_EQ(lagrange.center(), Coord(0, 0, h/4));
  EXPECT_EQ(lagrange.LocalToGlobal(1, 0, 0), Coord(a/2, 0, h/2));
  EXPECT_EQ(lagrange.LocalToGlobal(0, 1, 0), Coord(0, b/2, h/2));
  EXPECT_EQ(lagrange.LocalToGlobal(0, 0, 1), Coord(0, 0, h));
  for (int i_node = 0; i_node < lagrange.CountNodes(); ++i_node) {
    if (i_node == 4) {
      EXPECT_ANY_THROW(lagrange.GlobalToLocal(lagrange.GetGlobalCoord(i_node)));
      continue;
    }
    EXPECT_NEAR(0.0, (lagrange.GlobalToLocal(lagrange.GetGlobalCoord(i_node))
        - lagrange.GetLocalCoord(i_node)).norm(), 1e-15);
  }
  mini::geometry::Cell<typename Lagrange::Real> &cell = lagrange;
  // test the partition-of-unity property:
  std::srand(31415926);
  auto rand = [](){ return -1 + 2.0 * std::rand() / (1.0 + RAND_MAX); };
  for (int i = 0; i < 1000; ++i) {
    auto x = rand(), y = rand(), z = rand();
    auto shapes = cell.LocalToShapeFunctions(x, y, z);
    auto sum = std::accumulate(shapes.begin(), shapes.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-15);
    auto grads = cell.LocalToShapeGradients(x, y, z);
    Coord grads_sum
        = std::accumulate(grads.begin(), grads.end(), Coord(0, 0, 0));
    EXPECT_NEAR(grads_sum.norm(), 0.0, 1e-15);
    // compare gradients with O(h^2) finite difference derivatives
    int X{0}, Y{1}, Z{2};
    auto h = 1e-6;
    int n_node = cell.CountNodes();
    auto left = cell.LocalToShapeFunctions(x - h, y, z);
    auto right = cell.LocalToShapeFunctions(x + h, y, z);
    for (int i_node = 0; i_node < n_node; ++i_node) {
      grads[i_node][X] -= (right[i_node] - left[i_node]) / (2 * h);
    }
    left = cell.LocalToShapeFunctions(x, y - h, z);
    right = cell.LocalToShapeFunctions(x, y + h, z);
    for (int i_node = 0; i_node < n_node; ++i_node) {
      grads[i_node][Y] -= (right[i_node] - left[i_node]) / (2 * h);
    }
    left = cell.LocalToShapeFunctions(x, y, z - h);
    right = cell.LocalToShapeFunctions(x, y, z + h);
    for (int i_node = 0; i_node < n_node; ++i_node) {
      grads[i_node][Z] -= (right[i_node] - left[i_node]) / (2 * h);
      EXPECT_NEAR(grads[i_node].norm(), 0.0, 1e-9);
    }
  }
  // test the Kronecker-delta and property:
  for (int i = 0, n = cell.CountNodes(); i < n; ++i) {
    auto local_i = cell.GetLocalCoord(i);
    auto shapes = cell.LocalToShapeFunctions(local_i);
    for (int j = 0; j < n; ++j) {
      EXPECT_EQ(shapes[j], i == j);
    }
  }
  // test Jacobian determinant:
  auto abh = (a + a) * (b + b) * h;
  auto exact = [abh](double z){
    return (1 - z) * (1 - z) * abh / 32;
  };
  for (int i = 0; i < 1000; ++i) {
    auto x = rand(), y = rand(), z = rand();
    auto jacobian = cell.LocalToJacobian(x, y, z);
    EXPECT_NEAR(std::abs(jacobian.determinant()), exact(z), 1e-14);
  }
  // test consistency with shape functions on Triangle3's:
  {
    std::vector<std::vector<int>> c_lists{
        { 0, 1, 4 }, { 1, 2, 4 }, { 2, 3, 4 }, { 3, 0, 4 },
    };
    for (auto c_list : c_lists) {
      auto c2f = std::vector<int>(cell.CountNodes(), -1);
      int f = 0;
      for (int c : c_list) { c2f[c] = f++; }
      auto face = mini::geometry::Triangle3<double, 3>{
        cell.GetGlobalCoord(c_list[0]), cell.GetGlobalCoord(c_list[1]),
        cell.GetGlobalCoord(c_list[2]),
      };
      for (int i = 0; i < 1000; ++i) {
        auto x = (rand() + 1) / 2, y = (rand() + 1) / 2;
        auto global = face.LocalToGlobal(x, y);
        auto local = cell.GlobalToLocal(global);
        auto cell_shapes = cell.LocalToShapeFunctions(local);
        auto face_shapes = face.LocalToShapeFunctions(x, y);
        for (int c = 0; c < cell.CountNodes(); c++) {
          int f = c2f[c];
          EXPECT_NEAR(cell_shapes[c], f < 0 ? 0 : face_shapes[f], 1e-15);
        }
      }
    }
  }
  // test consistency with shape functions on Quadrangle4's:
  {
    std::vector<std::vector<int>> c_lists{ { 0, 3, 2, 1 } };
    for (auto c_list : c_lists) {
      auto c2f = std::vector<int>(cell.CountNodes(), -1);
      int f = 0;
      for (int c : c_list) { c2f[c] = f++; }
      auto face = mini::geometry::Quadrangle4<double, 3>{
        cell.GetGlobalCoord(c_list[0]), cell.GetGlobalCoord(c_list[1]),
        cell.GetGlobalCoord(c_list[2]), cell.GetGlobalCoord(c_list[3]),
      };
      for (int i = 0; i < 1000; ++i) {
        auto x = rand(), y = rand();
        auto global = face.LocalToGlobal(x, y);
        auto local = cell.GlobalToLocal(global);
        auto cell_shapes = cell.LocalToShapeFunctions(local);
        auto face_shapes = face.LocalToShapeFunctions(x, y);
        for (int c = 0; c < cell.CountNodes(); c++) {
          int f = c2f[c];
          EXPECT_NEAR(cell_shapes[c], f < 0 ? 0 : face_shapes[f], 1e-15);
        }
      }
    }
  }
}
TEST_F(TestLagrangePyramid5, SortNodesOnFace) {
  using mini::geometry::SortNodesOnFace;
  auto cell = Lagrange{
    Coord(-2, -2, 0), Coord(2, -2, 0), Coord(2, 2, 0), Coord(-2, 2, 0),
    Coord(0, 0, 4)
  };
  // test the version without conversion:
  {
    using Vector = std::vector<size_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55 }, face_nodes, face_nodes_expect;
    int face_n_node = 3;
    face_nodes = { 55, 11, 22, 0 };
    face_nodes_expect = { 11, 22, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 22, 33, 0 };
    face_nodes_expect = { 22, 33, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 33, 44, 0 };
    face_nodes_expect = { 33, 44, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 44, 11, 0 };
    face_nodes_expect = { 44, 11, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_n_node = 4;
    face_nodes = { 22, 33, 44, 11, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
  // test the version with conversion:
  {
    using Vector = std::vector<int16_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55 }, face_nodes, face_nodes_expect;
    int face_n_node = 3;
    face_nodes = { 55, 11, 22, 0 };
    face_nodes_expect = { 11, 22, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 22, 33, 0 };
    face_nodes_expect = { 22, 33, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 33, 44, 0 };
    face_nodes_expect = { 33, 44, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 44, 11, 0 };
    face_nodes_expect = { 44, 11, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_n_node = 4;
    face_nodes = { 22, 33, 44, 11, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
}

class TestLagrangePyramid13 : public ::testing::Test {
 protected:
  using Lagrange = mini::geometry::Pyramid13<double>;
  using Coord = typename Lagrange::Global;
};
TEST_F(TestLagrangePyramid13, CoordinateMap) {
  auto a = 2.0, b = 3.0, h = 4.0;
  auto lagrange = Lagrange{
    Coord(-a, -b, 0), Coord(+a, -b, 0), Coord(+a, +b, 0), Coord(-a, +b, 0),
    Coord(0, 0, h),
    Coord(0, -b, 0), Coord(+a, 0, 0), Coord(0, +b, 0), Coord(-a, 0, 0),
    Coord(-a/2, -b/2, h/2), Coord(+a/2, -b/2, h/2),
    Coord(+a/2, +b/2, h/2), Coord(-a/2, +b/2, h/2),
  };
  static_assert(lagrange.CellDim() == 3);
  static_assert(lagrange.PhysDim() == 3);
  EXPECT_EQ(lagrange.CountCorners(), 5);
  EXPECT_EQ(lagrange.CountNodes(), 13);
  EXPECT_EQ(lagrange.center(), Coord(0, 0, h/4));
  EXPECT_EQ(lagrange.LocalToGlobal(1, 0, 0), Coord(a/2, 0, h/2));
  EXPECT_EQ(lagrange.LocalToGlobal(0, 1, 0), Coord(0, b/2, h/2));
  EXPECT_EQ(lagrange.LocalToGlobal(0, 0, 1), Coord(0, 0, h));
  for (int i_node = 0; i_node < lagrange.CountNodes(); ++i_node) {
    if (i_node == 4) {
      EXPECT_ANY_THROW(lagrange.GlobalToLocal(lagrange.GetGlobalCoord(i_node)));
      continue;
    }
    EXPECT_NEAR(0.0, (lagrange.GlobalToLocal(lagrange.GetGlobalCoord(i_node))
        - lagrange.GetLocalCoord(i_node)).norm(), 1e-15);
  }
  mini::geometry::Cell<typename Lagrange::Real> &cell = lagrange;
  // test the partition-of-unity property:
  std::srand(31415926);
  auto rand = [](){ return -1 + 2.0 * std::rand() / (1.0 + RAND_MAX); };
  for (int i = 0; i < 1000; ++i) {
    auto x = rand(), y = rand(), z = rand();
    auto shapes = cell.LocalToShapeFunctions(x, y, z);
    auto sum = std::accumulate(shapes.begin(), shapes.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-15);
    auto grads = cell.LocalToShapeGradients(x, y, z);
    Coord grads_sum
        = std::accumulate(grads.begin(), grads.end(), Coord(0, 0, 0));
    EXPECT_NEAR(grads_sum.norm(), 0.0, 1e-15);
    // compare gradients with O(h^2) finite difference derivatives
    int X{0}, Y{1}, Z{2};
    auto h = 1e-6;
    int n_node = cell.CountNodes();
    auto left = cell.LocalToShapeFunctions(x - h, y, z);
    auto right = cell.LocalToShapeFunctions(x + h, y, z);
    for (int i_node = 0; i_node < n_node; ++i_node) {
      grads[i_node][X] -= (right[i_node] - left[i_node]) / (2 * h);
    }
    left = cell.LocalToShapeFunctions(x, y - h, z);
    right = cell.LocalToShapeFunctions(x, y + h, z);
    for (int i_node = 0; i_node < n_node; ++i_node) {
      grads[i_node][Y] -= (right[i_node] - left[i_node]) / (2 * h);
    }
    left = cell.LocalToShapeFunctions(x, y, z - h);
    right = cell.LocalToShapeFunctions(x, y, z + h);
    for (int i_node = 0; i_node < n_node; ++i_node) {
      grads[i_node][Z] -= (right[i_node] - left[i_node]) / (2 * h);
      EXPECT_NEAR(grads[i_node].norm(), 0.0, 1e-9);
    }
  }
  // test the Kronecker-delta and property:
  for (int i = 0, n = cell.CountNodes(); i < n; ++i) {
    auto local_i = cell.GetLocalCoord(i);
    auto shapes = cell.LocalToShapeFunctions(local_i);
    for (int j = 0; j < n; ++j) {
      EXPECT_EQ(shapes[j], i == j);
    }
  }
  // test Jacobian determinant:
  auto abh = (a + a) * (b + b) * h;
  auto exact = [abh](double z){
    return (1 - z) * (1 - z) * abh / 32;
  };
  for (int i = 0; i < 1000; ++i) {
    auto x = rand(), y = rand(), z = rand();
    auto jacobian = cell.LocalToJacobian(x, y, z);
    EXPECT_NEAR(std::abs(jacobian.determinant()), exact(z), 1e-14);
  }
  // test consistency with shape functions on Triangle6's:
  {
    std::vector<std::vector<int>> c_lists{
        { 0, 1, 4, 5, 10, 9 }, { 1, 2, 4, 6, 11, 10 },
        { 2, 3, 4, 7, 12, 11 }, { 3, 0, 4, 8, 9, 12 },
    };
    for (auto c_list : c_lists) {
      auto c2f = std::vector<int>(cell.CountNodes(), -1);
      int f = 0;
      for (int c : c_list) { c2f[c] = f++; }
      auto face = mini::geometry::Triangle6<double, 3>{
        cell.GetGlobalCoord(c_list[0]), cell.GetGlobalCoord(c_list[1]),
        cell.GetGlobalCoord(c_list[2]), cell.GetGlobalCoord(c_list[3]),
        cell.GetGlobalCoord(c_list[4]), cell.GetGlobalCoord(c_list[5]),
      };
      for (int i = 0; i < 1000; ++i) {
        auto x = (rand() + 1) / 2, y = (rand() + 1) / 2;
        auto global = face.LocalToGlobal(x, y);
        auto local = cell.GlobalToLocal(global);
        auto cell_shapes = cell.LocalToShapeFunctions(local);
        auto face_shapes = face.LocalToShapeFunctions(x, y);
        for (int c = 0; c < cell.CountNodes(); c++) {
          int f = c2f[c];
          EXPECT_NEAR(cell_shapes[c], f < 0 ? 0 : face_shapes[f], 1e-14);
        }
      }
    }
  }
  // test consistency with shape functions on Quadrangle8's:
  {
    std::vector<std::vector<int>> c_lists{
        { 0, 3, 2, 1, 8, 7, 6, 5 },
    };
    for (auto c_list : c_lists) {
      auto c2f = std::vector<int>(cell.CountNodes(), -1);
      int f = 0;
      for (int c : c_list) { c2f[c] = f++; }
      auto face = mini::geometry::Quadrangle8<double, 3>{
        cell.GetGlobalCoord(c_list[0]), cell.GetGlobalCoord(c_list[1]),
        cell.GetGlobalCoord(c_list[2]), cell.GetGlobalCoord(c_list[3]),
        cell.GetGlobalCoord(c_list[4]), cell.GetGlobalCoord(c_list[5]),
        cell.GetGlobalCoord(c_list[6]), cell.GetGlobalCoord(c_list[7]),
      };
      for (int i = 0; i < 1000; ++i) {
        auto x = rand(), y = rand();
        auto global = face.LocalToGlobal(x, y);
        auto local = cell.GlobalToLocal(global);
        auto cell_shapes = cell.LocalToShapeFunctions(local);
        auto face_shapes = face.LocalToShapeFunctions(x, y);
        for (int c = 0; c < cell.CountNodes(); c++) {
          int f = c2f[c];
          EXPECT_NEAR(cell_shapes[c], f < 0 ? 0 : face_shapes[f], 1e-15);
        }
      }
    }
  }
}
TEST_F(TestLagrangePyramid13, SortNodesOnFace) {
  using mini::geometry::SortNodesOnFace;
  auto a = 2.0, b = 3.0, h = 4.0;
  auto cell = Lagrange{
    Coord(-a, -b, 0), Coord(+a, -b, 0), Coord(+a, +b, 0), Coord(-a, +b, 0),
    Coord(0, 0, h),
    Coord(0, -b, 0), Coord(+a, 0, 0), Coord(0, +b, 0), Coord(-a, 0, 0),
    Coord(-a/2, -b/2, h/2), Coord(+a/2, -b/2, h/2),
    Coord(+a/2, +b/2, h/2), Coord(-a/2, +b/2, h/2),
  };
  // test the version without conversion:
  {
    using Vector = std::vector<size_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 1010,
      1111, 1212, 1313, 00 };
    Vector face_nodes, face_nodes_expect;
    int face_n_node = 6;
    face_nodes = { 55, 11, 22, 66, 1010, 1111, 0 };
    face_nodes_expect = { 11, 22, 55, 66, 1111, 1010, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 22, 33, 1212, 1111, 77, 0 };
    face_nodes_expect = { 22, 33, 55, 77, 1212, 1111, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 33, 44, 88, 1212, 1313, 0 };
    face_nodes_expect = { 33, 44, 55, 88, 1313, 1212, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 44, 11, 1010, 1313, 99, 0 };
    face_nodes_expect = { 44, 11, 55, 99, 1010, 1313, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_n_node = 8;
    face_nodes = { 22, 33, 44, 11, 77, 88, 99, 66, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 99, 88, 77, 66, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
  // test the version with conversion:
  {
    using Vector = std::vector<int16_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 1010,
      1111, 1212, 1313, 00 };
    Vector face_nodes, face_nodes_expect;
    int face_n_node = 6;
    face_nodes = { 55, 11, 22, 66, 1010, 1111, 0 };
    face_nodes_expect = { 11, 22, 55, 66, 1111, 1010, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 22, 33, 1212, 1111, 77, 0 };
    face_nodes_expect = { 22, 33, 55, 77, 1212, 1111, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 33, 44, 88, 1212, 1313, 0 };
    face_nodes_expect = { 33, 44, 55, 88, 1313, 1212, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 44, 11, 1010, 1313, 99, 0 };
    face_nodes_expect = { 44, 11, 55, 99, 1010, 1313, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_n_node = 8;
    face_nodes = { 22, 33, 44, 11, 77, 88, 99, 66, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 99, 88, 77, 66, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
}


class TestLagrangePyramid14 : public ::testing::Test {
 protected:
  using Lagrange = mini::geometry::Pyramid14<double>;
  using Coord = typename Lagrange::Global;
};
TEST_F(TestLagrangePyramid14, CoordinateMap) {
  auto a = 2.0, b = 3.0, h = 4.0;
  auto lagrange = Lagrange{
    Coord(-a, -b, 0), Coord(+a, -b, 0), Coord(+a, +b, 0), Coord(-a, +b, 0),
    Coord(0, 0, h),
    Coord(0, -b, 0), Coord(+a, 0, 0), Coord(0, +b, 0), Coord(-a, 0, 0),
    Coord(-a/2, -b/2, h/2), Coord(+a/2, -b/2, h/2),
    Coord(+a/2, +b/2, h/2), Coord(-a/2, +b/2, h/2),
    Coord(0, 0, 0),
  };
  static_assert(lagrange.CellDim() == 3);
  static_assert(lagrange.PhysDim() == 3);
  EXPECT_EQ(lagrange.CountCorners(), 5);
  EXPECT_EQ(lagrange.CountNodes(), 14);
  EXPECT_EQ(lagrange.center(), Coord(0, 0, h/4));
  EXPECT_EQ(lagrange.LocalToGlobal(1, 0, 0), Coord(a/2, 0, h/2));
  EXPECT_EQ(lagrange.LocalToGlobal(0, 1, 0), Coord(0, b/2, h/2));
  EXPECT_EQ(lagrange.LocalToGlobal(0, 0, 1), Coord(0, 0, h));
  for (int i_node = 0; i_node < lagrange.CountNodes(); ++i_node) {
    if (i_node == 4) {
      EXPECT_ANY_THROW(lagrange.GlobalToLocal(lagrange.GetGlobalCoord(i_node)));
      continue;
    }
    EXPECT_NEAR(0.0, (lagrange.GlobalToLocal(lagrange.GetGlobalCoord(i_node))
        - lagrange.GetLocalCoord(i_node)).norm(), 1e-15);
  }
  mini::geometry::Cell<typename Lagrange::Real> &cell = lagrange;
  // test the partition-of-unity property:
  std::srand(31415926);
  auto rand = [](){ return -1 + 2.0 * std::rand() / (1.0 + RAND_MAX); };
  for (int i = 0; i < 1000; ++i) {
    auto x = rand(), y = rand(), z = rand();
    auto shapes = cell.LocalToShapeFunctions(x, y, z);
    auto sum = std::accumulate(shapes.begin(), shapes.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-15);
    auto grads = cell.LocalToShapeGradients(x, y, z);
    Coord grads_sum
        = std::accumulate(grads.begin(), grads.end(), Coord(0, 0, 0));
    EXPECT_NEAR(grads_sum.norm(), 0.0, 1e-15);
    // compare gradients with O(h^2) finite difference derivatives
    int X{0}, Y{1}, Z{2};
    auto h = 1e-6;
    int n_node = cell.CountNodes();
    auto left = cell.LocalToShapeFunctions(x - h, y, z);
    auto right = cell.LocalToShapeFunctions(x + h, y, z);
    for (int i_node = 0; i_node < n_node; ++i_node) {
      grads[i_node][X] -= (right[i_node] - left[i_node]) / (2 * h);
    }
    left = cell.LocalToShapeFunctions(x, y - h, z);
    right = cell.LocalToShapeFunctions(x, y + h, z);
    for (int i_node = 0; i_node < n_node; ++i_node) {
      grads[i_node][Y] -= (right[i_node] - left[i_node]) / (2 * h);
    }
    left = cell.LocalToShapeFunctions(x, y, z - h);
    right = cell.LocalToShapeFunctions(x, y, z + h);
    for (int i_node = 0; i_node < n_node; ++i_node) {
      grads[i_node][Z] -= (right[i_node] - left[i_node]) / (2 * h);
      EXPECT_NEAR(grads[i_node].norm(), 0.0, 1e-9);
    }
  }
  // test the Kronecker-delta and property:
  for (int i = 0, n = cell.CountNodes(); i < n; ++i) {
    auto local_i = cell.GetLocalCoord(i);
    auto shapes = cell.LocalToShapeFunctions(local_i);
    for (int j = 0; j < n; ++j) {
      EXPECT_EQ(shapes[j], i == j);
    }
  }
  // test Jacobian determinant:
  auto abh = (a + a) * (b + b) * h;
  auto exact = [abh](double z){
    return (1 - z) * (1 - z) * abh / 32;
  };
  for (int i = 0; i < 1000; ++i) {
    auto x = rand(), y = rand(), z = rand();
    auto jacobian = cell.LocalToJacobian(x, y, z);
    EXPECT_NEAR(std::abs(jacobian.determinant()), exact(z), 1e-14);
  }
  // test consistency with shape functions on Triangle6's:
  {
    std::vector<std::vector<int>> c_lists{
        { 0, 1, 4, 5, 10, 9 }, { 1, 2, 4, 6, 11, 10 },
        { 2, 3, 4, 7, 12, 11 }, { 3, 0, 4, 8, 9, 12 },
    };
    for (auto c_list : c_lists) {
      auto c2f = std::vector<int>(cell.CountNodes(), -1);
      int f = 0;
      for (int c : c_list) { c2f[c] = f++; }
      auto face = mini::geometry::Triangle6<double, 3>{
        cell.GetGlobalCoord(c_list[0]), cell.GetGlobalCoord(c_list[1]),
        cell.GetGlobalCoord(c_list[2]), cell.GetGlobalCoord(c_list[3]),
        cell.GetGlobalCoord(c_list[4]), cell.GetGlobalCoord(c_list[5]),
      };
      for (int i = 0; i < 1000; ++i) {
        auto x = (rand() + 1) / 2, y = (rand() + 1) / 2;
        auto global = face.LocalToGlobal(x, y);
        auto local = cell.GlobalToLocal(global);
        auto cell_shapes = cell.LocalToShapeFunctions(local);
        auto face_shapes = face.LocalToShapeFunctions(x, y);
        for (int c = 0; c < cell.CountNodes(); c++) {
          int f = c2f[c];
          EXPECT_NEAR(cell_shapes[c], f < 0 ? 0 : face_shapes[f], 1e-14);
        }
      }
    }
  }
  // test consistency with shape functions on Quadrangle8's:
  {
    std::vector<std::vector<int>> c_lists{
        { 0, 3, 2, 1, 8, 7, 6, 5, 13 },
    };
    for (auto c_list : c_lists) {
      auto c2f = std::vector<int>(cell.CountNodes(), -1);
      int f = 0;
      for (int c : c_list) { c2f[c] = f++; }
      auto face = mini::geometry::Quadrangle9<double, 3>{
        cell.GetGlobalCoord(c_list[0]), cell.GetGlobalCoord(c_list[1]),
        cell.GetGlobalCoord(c_list[2]), cell.GetGlobalCoord(c_list[3]),
        cell.GetGlobalCoord(c_list[4]), cell.GetGlobalCoord(c_list[5]),
        cell.GetGlobalCoord(c_list[6]), cell.GetGlobalCoord(c_list[7]),
        cell.GetGlobalCoord(c_list[8]),
      };
      for (int i = 0; i < 1000; ++i) {
        auto x = rand(), y = rand();
        auto global = face.LocalToGlobal(x, y);
        auto local = cell.GlobalToLocal(global);
        auto cell_shapes = cell.LocalToShapeFunctions(local);
        auto face_shapes = face.LocalToShapeFunctions(x, y);
        for (int c = 0; c < cell.CountNodes(); c++) {
          int f = c2f[c];
          EXPECT_NEAR(cell_shapes[c], f < 0 ? 0 : face_shapes[f], 1e-15);
        }
      }
    }
  }
}
TEST_F(TestLagrangePyramid14, SortNodesOnFace) {
  using mini::geometry::SortNodesOnFace;
  auto a = 2.0, b = 3.0, h = 4.0;
  auto cell = Lagrange{
    Coord(-a, -b, 0), Coord(+a, -b, 0), Coord(+a, +b, 0), Coord(-a, +b, 0),
    Coord(0, 0, h),
    Coord(0, -b, 0), Coord(+a, 0, 0), Coord(0, +b, 0), Coord(-a, 0, 0),
    Coord(-a/2, -b/2, h/2), Coord(+a/2, -b/2, h/2),
    Coord(+a/2, +b/2, h/2), Coord(-a/2, +b/2, h/2),
    Coord(0, 0, 0),
  };
  // test the version without conversion:
  {
    using Vector = std::vector<size_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 1010,
      1111, 1212, 1313, 1414, 00 };
    Vector face_nodes, face_nodes_expect;
    int face_n_node = 6;
    face_nodes = { 55, 11, 22, 66, 1010, 1111, 0 };
    face_nodes_expect = { 11, 22, 55, 66, 1111, 1010, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 22, 33, 1212, 1111, 77, 0 };
    face_nodes_expect = { 22, 33, 55, 77, 1212, 1111, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 33, 44, 88, 1212, 1313, 0 };
    face_nodes_expect = { 33, 44, 55, 88, 1313, 1212, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 44, 11, 1010, 1313, 99, 0 };
    face_nodes_expect = { 44, 11, 55, 99, 1010, 1313, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_n_node = 9;
    face_nodes = { 1414, 22, 33, 44, 11, 77, 88, 99, 66, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 99, 88, 77, 66, 1414, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
  // test the version with conversion:
  {
    using Vector = std::vector<int16_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 1010,
      1111, 1212, 1313, 1414, 00 };
    Vector face_nodes, face_nodes_expect;
    int face_n_node = 6;
    face_nodes = { 55, 11, 22, 66, 1010, 1111, 0 };
    face_nodes_expect = { 11, 22, 55, 66, 1111, 1010, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 22, 33, 1212, 1111, 77, 0 };
    face_nodes_expect = { 22, 33, 55, 77, 1212, 1111, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 33, 44, 88, 1212, 1313, 0 };
    face_nodes_expect = { 33, 44, 55, 88, 1313, 1212, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 44, 11, 1010, 1313, 99, 0 };
    face_nodes_expect = { 44, 11, 55, 99, 1010, 1313, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_n_node = 9;
    face_nodes = { 1414, 22, 33, 44, 11, 77, 88, 99, 66, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 99, 88, 77, 66, 1414, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
