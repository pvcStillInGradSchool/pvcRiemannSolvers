//  Copyright 2023 PEI Weicheng

#include <cstdlib>

#include <algorithm>
#include <numeric>
#include <vector>

#include "mini/coordinate/cell.hpp"
#include "mini/coordinate/tetrahedron.hpp"

#include "gtest/gtest.h"

class TestCoordinateTetrahedron4 : public ::testing::Test {
 protected:
  using Coordinate = mini::coordinate::Tetrahedron4<double>;
  using Coord = typename Coordinate::Global;
};
TEST_F(TestCoordinateTetrahedron4, CoordinateMap) {
  auto tetra = Coordinate{
    Coord(0, 0, 0), Coord(3, 0, 0), Coord(0, 3, 0), Coord(0, 0, 3)
  };
  static_assert(tetra.CellDim() == 3);
  static_assert(tetra.PhysDim() == 3);
  EXPECT_EQ(tetra.CountCorners(), 4);
  EXPECT_EQ(tetra.CountNodes(), 4);
  EXPECT_EQ(tetra.center(), Coord(0.75, 0.75, 0.75));
  EXPECT_EQ(tetra.LocalToGlobal(1, 0, 0), Coord(0, 0, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 1, 0), Coord(3, 0, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 0, 1), Coord(0, 3, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 0, 0), Coord(0, 0, 3));
  EXPECT_EQ(tetra.GlobalToLocal(0, 0, 0), Coord(1, 0, 0));
  EXPECT_EQ(tetra.GlobalToLocal(3, 0, 0), Coord(0, 1, 0));
  EXPECT_EQ(tetra.GlobalToLocal(0, 3, 0), Coord(0, 0, 1));
  EXPECT_EQ(tetra.GlobalToLocal(0, 0, 3), Coord(0, 0, 0));
  mini::coordinate::Cell<typename Coordinate::Real> &cell = tetra;
  // test the partition-of-unity property:
  std::srand(31415926);
  auto rand = [](){ return std::rand() / (1.0 + RAND_MAX); };
  for (int i = 0; i < 1000; ++i) {
    auto x = rand(), y = rand(), z = rand();
    auto shapes = cell.LocalToShapeFunctions(x, y, z);
    auto sum = std::accumulate(shapes.begin(), shapes.end(), 0.0);
    EXPECT_EQ(sum, 1.0);
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
    auto local_i = cell.GetLocal(i);
    auto shapes = cell.LocalToShapeFunctions(local_i);
    for (int j = 0; j < n; ++j) {
      EXPECT_EQ(shapes[j], i == j);
    }
  }
}
TEST_F(TestCoordinateTetrahedron4, SortNodesOnFace) {
  using mini::coordinate::SortNodesOnFace;
  auto cell = Coordinate{
    Coord(0, 0, 0), Coord(3, 0, 0), Coord(0, 3, 0), Coord(0, 0, 3)
  };
  int face_n_node = 3;
  // test the version without conversion:
  {
    using Vector = std::vector<size_t>;
    Vector cell_nodes{ 11, 22, 33, 44 }, face_nodes, face_nodes_expect;
    face_nodes = { 11, 22, 33, 0 };
    face_nodes_expect = { 11, 33, 22, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 44, 22, 0 };
    face_nodes_expect = { 11, 22, 44, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 33, 44, 0 };
    face_nodes_expect = { 11, 44, 33, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 44, 0 };
    face_nodes_expect = { 22, 33, 44, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
  // test the version with conversion:
  {
    using Vector = std::vector<int16_t>;
    Vector cell_nodes{ 11, 22, 33, 44 }, face_nodes, face_nodes_expect;
    face_nodes = { 11, 22, 33, 0 };
    face_nodes_expect = { 11, 33, 22, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 44, 22, 0 };
    face_nodes_expect = { 11, 22, 44, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 33, 44, 0 };
    face_nodes_expect = { 11, 44, 33, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 44, 0 };
    face_nodes_expect = { 22, 33, 44, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
}

class TestCoordinateTetrahedron10 : public ::testing::Test {
 protected:
  using Coordinate = mini::coordinate::Tetrahedron10<double>;
  using Coord = typename Coordinate::Global;
};
TEST_F(TestCoordinateTetrahedron10, CoordinateMap) {
  auto tetra = Coordinate{
    Coord(0, 0, 0), Coord(6, 0, 0), Coord(0, 6, 0), Coord(0, 0, 6),
    Coord(3, 0, 0), Coord(3, 3, 0), Coord(0, 3, 0),
    Coord(0, 0, 3), Coord(3, 0, 3), Coord(0, 3, 3),
  };
  static_assert(tetra.CellDim() == 3);
  static_assert(tetra.PhysDim() == 3);
  EXPECT_EQ(tetra.CountCorners(), 4);
  EXPECT_EQ(tetra.CountNodes(), 10);
  EXPECT_EQ(tetra.center(), Coord(1.5, 1.5, 1.5));
  EXPECT_EQ(tetra.LocalToGlobal(1, 0, 0), Coord(0, 0, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 1, 0), Coord(6, 0, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 0, 1), Coord(0, 6, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 0, 0), Coord(0, 0, 6));
  EXPECT_EQ(tetra.GlobalToLocal(0, 0, 0), Coord(1, 0, 0));
  EXPECT_EQ(tetra.GlobalToLocal(6, 0, 0), Coord(0, 1, 0));
  EXPECT_EQ(tetra.GlobalToLocal(0, 6, 0), Coord(0, 0, 1));
  EXPECT_EQ(tetra.GlobalToLocal(0, 0, 6), Coord(0, 0, 0));
  mini::coordinate::Cell<typename Coordinate::Real> &cell = tetra;
  // test the partition-of-unity property:
  std::srand(31415926);
  auto rand = [](){ return std::rand() / (1.0 + RAND_MAX); };
  for (int i = 0; i < 1000; ++i) {
    auto x = rand(), y = rand(), z = rand();
    auto shapes = cell.LocalToShapeFunctions(x, y, z);
    auto sum = std::accumulate(shapes.begin(), shapes.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-14);
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
      EXPECT_NEAR(grads[i_node].norm(), 0.0, 1e-8);
    }
  }
  // test the Kronecker-delta and property:
  for (int i = 0, n = cell.CountNodes(); i < n; ++i) {
    auto local_i = cell.GetLocal(i);
    auto shapes = cell.LocalToShapeFunctions(local_i);
    for (int j = 0; j < n; ++j) {
      EXPECT_EQ(shapes[j], i == j);
    }
  }
}
TEST_F(TestCoordinateTetrahedron10, SortNodesOnFace) {
  using mini::coordinate::SortNodesOnFace;
  auto cell = Coordinate{
    Coord(0, 0, 0), Coord(6, 0, 0), Coord(0, 6, 0), Coord(0, 0, 6),
    Coord(3, 0, 0), Coord(3, 3, 0), Coord(0, 3, 0),
    Coord(0, 0, 3), Coord(3, 0, 3), Coord(0, 3, 3),
  };
  int face_n_node = 6;
  // test the version without conversion:
  {
    using Vector = std::vector<size_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 1010, 0 };
    Vector face_nodes, face_nodes_expect;
    face_nodes = { 11, 22, 33, 55, 66, 77, 0 };
    face_nodes_expect = { 11, 33, 22, 77, 66, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 44, 22, 55, 88, 99, 0 };
    face_nodes_expect = { 11, 22, 44, 55, 99, 88, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 33, 44, 1010, 88, 77, 0 };
    face_nodes_expect = { 11, 44, 33, 88, 1010, 77, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 44, 1010, 99, 66, 0 };
    face_nodes_expect = { 22, 33, 44, 66, 1010, 99, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
  // test the version with conversion:
  {
    using Vector = std::vector<int16_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 1010, 0 };
    Vector face_nodes, face_nodes_expect;
    face_nodes = { 11, 22, 33, 55, 66, 77, 0 };
    face_nodes_expect = { 11, 33, 22, 77, 66, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 44, 22, 55, 88, 99, 0 };
    face_nodes_expect = { 11, 22, 44, 55, 99, 88, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 33, 44, 1010, 88, 77, 0 };
    face_nodes_expect = { 11, 44, 33, 88, 1010, 77, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 44, 1010, 99, 66, 0 };
    face_nodes_expect = { 22, 33, 44, 66, 1010, 99, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
