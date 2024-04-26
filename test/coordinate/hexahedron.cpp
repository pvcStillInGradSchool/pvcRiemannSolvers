//  Copyright 2023 PEI Weicheng

#include <numeric>

#include <cmath>

#include "mini/geometry/cell.hpp"
#include "mini/geometry/hexahedron.hpp"
#include "mini/constant/index.hpp"

#include "gtest/gtest.h"

using namespace mini::constant::index;

double rand_f() {
  return -1 + 2.0 * std::rand() / (1.0 + RAND_MAX);
}

class TestCoordinateHexahedron8 : public ::testing::Test {
 protected:
  using Coordinate = mini::geometry::Hexahedron8<double>;
  using Coord = typename Coordinate::Global;
  using Global = Coord;
  using Local = Coord;
  using Gradient = Coord;
};
TEST_F(TestCoordinateHexahedron8, CoordinateMap) {
  auto hexa = Coordinate {
    Coord(-10, -10, -10), Coord(+10, -10, -10),
    Coord(+10, +10, -10), Coord(-10, +10, -10),
    Coord(-10, -10, +10), Coord(+10, -10, +10),
    Coord(+10, +10, +10), Coord(-10, +10, +10)
  };
  static_assert(hexa.CellDim() == 3);
  static_assert(hexa.PhysDim() == 3);
  EXPECT_EQ(hexa.CountCorners(), 8);
  EXPECT_EQ(hexa.CountNodes(), 8);
  EXPECT_EQ(hexa.LocalToGlobal(1, 1, 1), Coord(10, 10, 10));
  EXPECT_EQ(hexa.LocalToGlobal(1.5, 1.5, 1.5), Coord(15, 15, 15));
  EXPECT_EQ(hexa.LocalToGlobal(3, 4, 5), Coord(30, 40, 50));
  EXPECT_NEAR(0, (hexa.GlobalToLocal(30, 40, 20) - Coord(3, 4, 2)).norm(),
      1e-10);
  EXPECT_NEAR(0, (hexa.GlobalToLocal(40, 55, 25) - Coord(4, 5.5, 2.5)).norm(),
      1e-10);
  EXPECT_NEAR(0, (hexa.GlobalToLocal(70, 130, 60) - Coord(7, 13, 6)).norm(),
      1e-10);
  EXPECT_NEAR(0, (hexa.GlobalToLocal(-20, -10, -10) - Coord(-2, -1, -1)).norm(),
      1e-10);
  EXPECT_NEAR(0, (hexa.GlobalToLocal(-10, -20, -10) - Coord(-1, -2, -1)).norm(),
      1e-10);
  EXPECT_NEAR(0, (hexa.GlobalToLocal(-10, -10, -20) - Coord(-1, -1, -2)).norm(),
      1e-10);
  EXPECT_NEAR(0, (hexa.GlobalToLocal(+20, -10, -10) - Coord(+2, -1, -1)).norm(),
      1e-10);
  EXPECT_NEAR(0, (hexa.GlobalToLocal(-10, +20, -10) - Coord(-1, +2, -1)).norm(),
      1e-10);
  EXPECT_NEAR(0, (hexa.GlobalToLocal(-10, -10, +20) - Coord(-1, -1, +2)).norm(),
      1e-10);
  EXPECT_NEAR(0, (hexa.GlobalToLocal(-20, +10, +10) - Coord(-2, +1, +1)).norm(),
      1e-10);
  EXPECT_NEAR(0, (hexa.GlobalToLocal(+10, -20, +10) - Coord(+1, -2, +1)).norm(),
      1e-10);
  EXPECT_NEAR(0, (hexa.GlobalToLocal(+10, +10, -20) - Coord(+1, +1, -2)).norm(),
      1e-10);
  EXPECT_NEAR(0, (hexa.GlobalToLocal(+20, +10, +10) - Coord(+2, +1, +1)).norm(),
      1e-10);
  EXPECT_NEAR(0, (hexa.GlobalToLocal(+10, +20, +10) - Coord(+1, +2, +1)).norm(),
      1e-10);
  EXPECT_NEAR(0, (hexa.GlobalToLocal(+10, +10, +20) - Coord(+1, +1, +2)).norm(),
      1e-10);
  mini::geometry::Cell<typename Coordinate::Real> &cell = hexa;
  // test the partition-of-unity property:
  std::srand(31415926);
  for (int i = 0; i < 1000; ++i) {
    auto x = rand_f(), y = rand_f(), z = rand_f();
    auto shapes = cell.LocalToShapeFunctions(x, y, z);
    auto sum = std::accumulate(shapes.begin(), shapes.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-15);
    auto grads = cell.LocalToShapeGradients(x, y, z);
    Coord grads_sum
        = std::accumulate(grads.begin(), grads.end(), Coord(0, 0, 0));
    EXPECT_NEAR(grads_sum.norm(), 0.0, 1e-15);
    // compare gradients with O(h^2) finite difference derivatives
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
  hexa = Coordinate {
    Coord(2.00000, 1.000000, 0.0), Coord(1.94313, 0.878607, 0.0),
    Coord(2.00493, 0.845382, 0.0), Coord(2.06283, 0.874438, 0.0),
    Coord(2.00000, 1.000000, 0.1), Coord(1.94313, 0.878607, 0.1),
    Coord(2.00493, 0.845382, 0.1), Coord(2.06283, 0.874438, 0.1),
  };
  EXPECT_ANY_THROW(hexa.GlobalToLocal(2.05723, 0.777978, 0));
}
TEST_F(TestCoordinateHexahedron8, GetJacobianGradient) {
  std::srand(31415926);
  for (int i_cell = 1 << 5; i_cell > 0; --i_cell) {
    // build a hexa-gauss and a Lagrange basis on it
    auto a = 20.0, b = 30.0, c = 40.0;
    auto cell = Coordinate {
      Global(rand_f() - a, rand_f() - b, rand_f() - c),
      Global(rand_f() + a, rand_f() - b, rand_f() - c),
      Global(rand_f() + a, rand_f() + b, rand_f() - c),
      Global(rand_f() - a, rand_f() + b, rand_f() - c),
      Global(rand_f() - a, rand_f() - b, rand_f() + c),
      Global(rand_f() + a, rand_f() - b, rand_f() + c),
      Global(rand_f() + a, rand_f() + b, rand_f() + c),
      Global(rand_f() - a, rand_f() + b, rand_f() + c),
    };
    // compare gradients with O(h^2) finite difference derivatives
    for (int i_point = 1 << 5; i_point > 0; --i_point) {
      auto x = rand_f(), y = rand_f(), z = rand_f();
      auto local = Local{x, y, z};
      auto mat_grad = cell.LocalToJacobianGradient(local);
      auto det_grad = cell.LocalToJacobianDeterminantGradient(local);
      auto h = 1e-5;
      auto mat_left = cell.LocalToJacobian(x - h, y, z);
      auto mat_right = cell.LocalToJacobian(x + h, y, z);
      mat_grad[X] -= (mat_right - mat_left) / (2 * h);
      auto det_left = mat_left.determinant();
      auto det_right = mat_right.determinant();
      det_grad[X] -= (det_right - det_left) / (2 * h);
      EXPECT_NEAR(mat_grad[X].norm(), 0.0, 1e-8);
      mat_left = cell.LocalToJacobian(x, y - h, z);
      mat_right = cell.LocalToJacobian(x, y + h, z);
      mat_grad[Y] -= (mat_right - mat_left) / (2 * h);
      det_left = mat_left.determinant();
      det_right = mat_right.determinant();
      det_grad[Y] -= (det_right - det_left) / (2 * h);
      EXPECT_NEAR(mat_grad[Y].norm(), 0.0, 1e-8);
      mat_left = cell.LocalToJacobian(x, y, z - h);
      mat_right = cell.LocalToJacobian(x, y, z + h);
      mat_grad[Z] -= (mat_right - mat_left) / (2 * h);
      det_left = mat_left.determinant();
      det_right = mat_right.determinant();
      det_grad[Z] -= (det_right - det_left) / (2 * h);
      EXPECT_NEAR(mat_grad[Z].norm(), 0.0, 1e-8);
      auto det = cell.LocalToJacobian(x, y, z).determinant();
      EXPECT_NEAR(det_grad.norm(), 0.0, det * 1e-10);
    }
    for (int i_point = 1 << 5; i_point > 0; --i_point) {
      auto x = rand_f(), y = rand_f(), z = rand_f();
      auto local = Local{x, y, z};
      auto det = cell.LocalToJacobian(local).determinant();
      auto det_hess = cell.LocalToJacobianDeterminantHessian(local);
      auto h = 1e-5;
      Gradient det_grad_diff = (
          cell.LocalToJacobianDeterminantGradient(Local(x + h, y, z)) -
          cell.LocalToJacobianDeterminantGradient(Local(x - h, y, z))
      ) / (2 * h);
      EXPECT_NEAR(det_hess[XX], det_grad_diff[X], det * 1e-10);
      EXPECT_NEAR(det_hess[XY], det_grad_diff[Y], det * 1e-10);
      EXPECT_NEAR(det_hess[XZ], det_grad_diff[Z], det * 1e-10);
      det_grad_diff = (
          cell.LocalToJacobianDeterminantGradient(Local(x, y + h, z)) -
          cell.LocalToJacobianDeterminantGradient(Local(x, y - h, z))
      ) / (2 * h);
      EXPECT_NEAR(det_hess[YX], det_grad_diff[X], det * 1e-10);
      EXPECT_NEAR(det_hess[YY], det_grad_diff[Y], det * 1e-10);
      EXPECT_NEAR(det_hess[YZ], det_grad_diff[Z], det * 1e-10);
      det_grad_diff = (
          cell.LocalToJacobianDeterminantGradient(Local(x, y, z + h)) -
          cell.LocalToJacobianDeterminantGradient(Local(x, y, z - h))
      ) / (2 * h);
      EXPECT_NEAR(det_hess[ZX], det_grad_diff[X], det * 1e-10);
      EXPECT_NEAR(det_hess[ZY], det_grad_diff[Y], det * 1e-10);
      EXPECT_NEAR(det_hess[ZZ], det_grad_diff[Z], det * 1e-10);
    }
  }
}
TEST_F(TestCoordinateHexahedron8, SortNodesOnFace) {
  using mini::geometry::SortNodesOnFace;
  auto cell = Coordinate{
    Coord(-10, -10, -10), Coord(+10, -10, -10),
    Coord(+10, +10, -10), Coord(-10, +10, -10),
    Coord(-10, -10, +10), Coord(+10, -10, +10),
    Coord(+10, +10, +10), Coord(-10, +10, +10)
  };
  EXPECT_EQ(cell.GetOutwardNormalVector(0), Coord(0, 0, -1));
  EXPECT_EQ(cell.GetOutwardNormalVector(1), Coord(0, -1, 0));
  EXPECT_EQ(cell.GetOutwardNormalVector(2), Coord(+1, 0, 0));
  EXPECT_EQ(cell.GetOutwardNormalVector(3), Coord(0, +1, 0));
  EXPECT_EQ(cell.GetOutwardNormalVector(4), Coord(-1, 0, 0));
  EXPECT_EQ(cell.GetOutwardNormalVector(5), Coord(0, 0, +1));
  int face_n_node = 4;
  // test the version without conversion:
  {
    using Vector = std::vector<size_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 00 };
    Vector face_nodes, face_nodes_expect;
    face_nodes = { 11, 22, 33, 44, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 22, 55, 66, 0 };
    face_nodes_expect = { 11, 22, 66, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 44, 55, 88, 0 };
    face_nodes_expect = { 11, 55, 88, 44, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 66, 77, 0 };
    face_nodes_expect = { 22, 33, 77, 66, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 33, 44, 77, 88, 0 };
    face_nodes_expect = { 33, 44, 88, 77, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 66, 77, 88, 0 };
    face_nodes_expect = { 55, 66, 77, 88, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
  // test the version with conversion:
  {
    using Vector = std::vector<int16_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 00 };
    Vector face_nodes, face_nodes_expect;
    face_nodes = { 11, 22, 33, 44, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 22, 55, 66, 0 };
    face_nodes_expect = { 11, 22, 66, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 44, 55, 88, 0 };
    face_nodes_expect = { 11, 55, 88, 44, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 66, 77, 0 };
    face_nodes_expect = { 22, 33, 77, 66, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 33, 44, 77, 88, 0 };
    face_nodes_expect = { 33, 44, 88, 77, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 66, 77, 88, 0 };
    face_nodes_expect = { 55, 66, 77, 88, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
}

class TestCoordinateHexahedron20 : public ::testing::Test {
 protected:
  using Coordinate = mini::geometry::Hexahedron20<double>;
  using Coord = typename Coordinate::Global;
  using Global = Coord;
  using Local = Coord;
  using Gradient = Coord;
};
TEST_F(TestCoordinateHexahedron20, CoordinateMap) {
  auto hexa = Coordinate {
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
  static_assert(hexa.CellDim() == 3);
  static_assert(hexa.PhysDim() == 3);
  EXPECT_EQ(hexa.CountCorners(), 8);
  EXPECT_EQ(hexa.CountNodes(), 20);
  EXPECT_EQ(hexa.LocalToGlobal(1, 1, 1), Coord(10, 10, 10));
  EXPECT_EQ(hexa.LocalToGlobal(1.5, 1.5, 1.5), Coord(15, 15, 15));
  EXPECT_EQ(hexa.LocalToGlobal(3, 4, 5), Coord(30, 40, 50));
  EXPECT_NEAR(0, (hexa.GlobalToLocal(30, 40, 20) - Coord(3, 4, 2)).norm(),
      1e-10);
  EXPECT_NEAR(0, (hexa.GlobalToLocal(40, 55, 25) - Coord(4, 5.5, 2.5)).norm(),
      1e-10);
  EXPECT_NEAR(0, (hexa.GlobalToLocal(70, 130, 60) - Coord(7, 13, 6)).norm(),
      1e-10);
  mini::geometry::Cell<typename Coordinate::Real> &cell = hexa;
  // test the partition-of-unity property:
  std::srand(31415926);
  for (int i = 0; i < 1000; ++i) {
    auto x = rand_f(), y = rand_f(), z = rand_f();
    auto shapes = cell.LocalToShapeFunctions(x, y, z);
    auto sum = std::accumulate(shapes.begin(), shapes.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-15);
    auto grads = cell.LocalToShapeGradients(x, y, z);
    Coord grads_sum
        = std::accumulate(grads.begin(), grads.end(), Coord(0, 0, 0));
    EXPECT_NEAR(grads_sum.norm(), 0.0, 1e-15);
    // compare gradients with O(h^2) finite difference derivatives
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
}
TEST_F(TestCoordinateHexahedron20, GetJacobianGradient) {
  std::srand(31415926);
  for (int i_cell = 1 << 5; i_cell > 0; --i_cell) {
    // build a hexa-gauss and a Lagrange basis on it
    auto a = 20.0, b = 30.0, c = 40.0;
    auto cell = Coordinate {
      // corner nodes on the bottom face
      Global(rand_f() - a, rand_f() - b, rand_f() - c),
      Global(rand_f() + a, rand_f() - b, rand_f() - c),
      Global(rand_f() + a, rand_f() + b, rand_f() - c),
      Global(rand_f() - a, rand_f() + b, rand_f() - c),
      // corner nodes on the top face
      Global(rand_f() - a, rand_f() - b, rand_f() + c),
      Global(rand_f() + a, rand_f() - b, rand_f() + c),
      Global(rand_f() + a, rand_f() + b, rand_f() + c),
      Global(rand_f() - a, rand_f() + b, rand_f() + c),
      // mid-edge nodes on the bottom face
      Global(rand_f(), rand_f() - b, rand_f() - c),
      Global(rand_f() + a, rand_f(), rand_f() - c),
      Global(rand_f(), rand_f() + b, rand_f() - c),
      Global(rand_f() - a, rand_f(), rand_f() - c),
      // mid-edge nodes on vertical edges
      Global(rand_f() - a, rand_f() - b, rand_f()),
      Global(rand_f() + a, rand_f() - b, rand_f()),
      Global(rand_f() + a, rand_f() + b, rand_f()),
      Global(rand_f() - a, rand_f() + b, rand_f()),
      // mid-edge nodes on the top face
      Global(rand_f(), rand_f() - b, rand_f() + c),
      Global(rand_f() + a, rand_f(), rand_f() + c),
      Global(rand_f(), rand_f() + b, rand_f() + c),
      Global(rand_f() - a, rand_f(), rand_f() + c),
    };
    // compare gradients with O(h^2) finite difference derivatives
    for (int i_point = 1 << 5; i_point > 0; --i_point) {
      auto x = rand_f(), y = rand_f(), z = rand_f();
      auto local = Local{x, y, z};
      auto mat_grad = cell.LocalToJacobianGradient(local);
      auto det_grad = cell.LocalToJacobianDeterminantGradient(local);
      auto h = 1e-5;
      auto mat_left = cell.LocalToJacobian(x - h, y, z);
      auto mat_right = cell.LocalToJacobian(x + h, y, z);
      mat_grad[X] -= (mat_right - mat_left) / (2 * h);
      auto det_left = mat_left.determinant();
      auto det_right = mat_right.determinant();
      det_grad[X] -= (det_right - det_left) / (2 * h);
      EXPECT_NEAR(mat_grad[X].norm(), 0.0, 1e-8);
      mat_left = cell.LocalToJacobian(x, y - h, z);
      mat_right = cell.LocalToJacobian(x, y + h, z);
      mat_grad[Y] -= (mat_right - mat_left) / (2 * h);
      det_left = mat_left.determinant();
      det_right = mat_right.determinant();
      det_grad[Y] -= (det_right - det_left) / (2 * h);
      EXPECT_NEAR(mat_grad[Y].norm(), 0.0, 1e-8);
      mat_left = cell.LocalToJacobian(x, y, z - h);
      mat_right = cell.LocalToJacobian(x, y, z + h);
      mat_grad[Z] -= (mat_right - mat_left) / (2 * h);
      det_left = mat_left.determinant();
      det_right = mat_right.determinant();
      det_grad[Z] -= (det_right - det_left) / (2 * h);
      EXPECT_NEAR(mat_grad[Z].norm(), 0.0, 1e-8);
      auto det = cell.LocalToJacobian(x, y, z).determinant();
      EXPECT_NEAR(det_grad.norm(), 0.0, det * 1e-10);
    }
    for (int i_point = 1 << 5; i_point > 0; --i_point) {
      auto x = rand_f(), y = rand_f(), z = rand_f();
      auto local = Local{x, y, z};
      auto det = cell.LocalToJacobian(local).determinant();
      auto det_hess = cell.LocalToJacobianDeterminantHessian(local);
      auto h = 1e-5;
      Gradient det_grad_diff = (
          cell.LocalToJacobianDeterminantGradient(Local(x + h, y, z)) -
          cell.LocalToJacobianDeterminantGradient(Local(x - h, y, z))
      ) / (2 * h);
      EXPECT_NEAR(det_hess[XX], det_grad_diff[X], det * 1e-10);
      EXPECT_NEAR(det_hess[XY], det_grad_diff[Y], det * 1e-10);
      EXPECT_NEAR(det_hess[XZ], det_grad_diff[Z], det * 1e-10);
      det_grad_diff = (
          cell.LocalToJacobianDeterminantGradient(Local(x, y + h, z)) -
          cell.LocalToJacobianDeterminantGradient(Local(x, y - h, z))
      ) / (2 * h);
      EXPECT_NEAR(det_hess[YX], det_grad_diff[X], det * 1e-10);
      EXPECT_NEAR(det_hess[YY], det_grad_diff[Y], det * 1e-10);
      EXPECT_NEAR(det_hess[YZ], det_grad_diff[Z], det * 1e-10);
      det_grad_diff = (
          cell.LocalToJacobianDeterminantGradient(Local(x, y, z + h)) -
          cell.LocalToJacobianDeterminantGradient(Local(x, y, z - h))
      ) / (2 * h);
      EXPECT_NEAR(det_hess[ZX], det_grad_diff[X], det * 1e-10);
      EXPECT_NEAR(det_hess[ZY], det_grad_diff[Y], det * 1e-10);
      EXPECT_NEAR(det_hess[ZZ], det_grad_diff[Z], det * 1e-10);
    }
  }
}
TEST_F(TestCoordinateHexahedron20, SortNodesOnFace) {
  using mini::geometry::SortNodesOnFace;
  auto cell = Coordinate{
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
  EXPECT_EQ(cell.GetOutwardNormalVector(0), Coord(0, 0, -1));
  EXPECT_EQ(cell.GetOutwardNormalVector(1), Coord(0, -1, 0));
  EXPECT_EQ(cell.GetOutwardNormalVector(2), Coord(+1, 0, 0));
  EXPECT_EQ(cell.GetOutwardNormalVector(3), Coord(0, +1, 0));
  EXPECT_EQ(cell.GetOutwardNormalVector(4), Coord(-1, 0, 0));
  EXPECT_EQ(cell.GetOutwardNormalVector(5), Coord(0, 0, +1));
  int face_n_node = 8;
  // test the version without conversion:
  {
    using Vector = std::vector<size_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 1010,
      1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020, 00 };
    Vector face_nodes, face_nodes_expect;
    face_nodes = { 11, 22, 33, 44, 1111, 1010, 1212, 99, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 1212, 1111, 1010, 99, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 22, 55, 66, 1313, 1414, 1717, 99, 0 };
    face_nodes_expect = { 11, 22, 66, 55, 99, 1414, 1717, 1313, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 44, 55, 88, 1212, 1313, 2020, 1616, 0 };
    face_nodes_expect = { 11, 55, 88, 44, 1313, 2020, 1616, 1212, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 66, 77, 1818, 1515, 1414, 1010, 0 };
    face_nodes_expect = { 22, 33, 77, 66, 1010, 1515, 1818, 1414, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 33, 44, 77, 88, 1919, 1616, 1515, 1111, 0 };
    face_nodes_expect = { 33, 44, 88, 77, 1111, 1616, 1919, 1515, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 66, 77, 88, 2020, 1919, 1818, 1717, 0 };
    face_nodes_expect = { 55, 66, 77, 88, 1717, 1818, 1919, 2020, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
  // test the version with conversion:
  {
    using Vector = std::vector<int16_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 1010,
      1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020, 00 };
    Vector face_nodes, face_nodes_expect;
    face_nodes = { 11, 22, 33, 44, 1111, 1010, 1212, 99, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 1212, 1111, 1010, 99, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 22, 55, 66, 1313, 1414, 1717, 99, 0 };
    face_nodes_expect = { 11, 22, 66, 55, 99, 1414, 1717, 1313, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 44, 55, 88, 1212, 1313, 2020, 1616, 0 };
    face_nodes_expect = { 11, 55, 88, 44, 1313, 2020, 1616, 1212, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 66, 77, 1818, 1515, 1414, 1010, 0 };
    face_nodes_expect = { 22, 33, 77, 66, 1010, 1515, 1818, 1414, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 33, 44, 77, 88, 1919, 1616, 1515, 1111, 0 };
    face_nodes_expect = { 33, 44, 88, 77, 1111, 1616, 1919, 1515, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 66, 77, 88, 2020, 1919, 1818, 1717, 0 };
    face_nodes_expect = { 55, 66, 77, 88, 1717, 1818, 1919, 2020, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
}

class TestCoordinateHexahedron27 : public ::testing::Test {
 protected:
  using Coordinate = mini::geometry::Hexahedron27<double>;
  using Coord = typename Coordinate::Global;
  using Global = Coord;
  using Local = Coord;
  using Gradient = Coord;
};
TEST_F(TestCoordinateHexahedron27, CoordinateMap) {
  auto hexa = Coordinate {
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
    Coord(0, -10, 0), Coord(+10, 0, 0),
    Coord(0, +10, 0), Coord(-10, 0, 0),
    Coord(0, 0, +10),
    // center node
    Coord(0, 0, 0),
  };
  static_assert(hexa.CellDim() == 3);
  static_assert(hexa.PhysDim() == 3);
  EXPECT_EQ(hexa.CountCorners(), 8);
  EXPECT_EQ(hexa.CountNodes(), 27);
  EXPECT_EQ(hexa.LocalToGlobal(1, 1, 1), Coord(10, 10, 10));
  EXPECT_EQ(hexa.LocalToGlobal(1.5, 1.5, 1.5), Coord(15, 15, 15));
  EXPECT_EQ(hexa.LocalToGlobal(3, 4, 5), Coord(30, 40, 50));
  EXPECT_NEAR(0, (hexa.GlobalToLocal(30, 40, 20) - Coord(3, 4, 2)).norm(),
      1e-10);
  EXPECT_NEAR(0, (hexa.GlobalToLocal(40, 55, 25) - Coord(4, 5.5, 2.5)).norm(),
      1e-10);
  EXPECT_NEAR(0, (hexa.GlobalToLocal(70, 130, 60) - Coord(7, 13, 6)).norm(),
      1e-10);
  mini::geometry::Cell<typename Coordinate::Real> &cell = hexa;
  // test the partition-of-unity property:
  std::srand(31415926);
  for (int i = 0; i < 1000; ++i) {
    auto x = rand_f(), y = rand_f(), z = rand_f();
    auto shapes = cell.LocalToShapeFunctions(x, y, z);
    auto sum = std::accumulate(shapes.begin(), shapes.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-15);
    auto grads = cell.LocalToShapeGradients(x, y, z);
    Coord grads_sum
        = std::accumulate(grads.begin(), grads.end(), Coord(0, 0, 0));
    EXPECT_NEAR(grads_sum.norm(), 0.0, 1e-15);
    // compare gradients with O(h^2) finite difference derivatives
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
}
TEST_F(TestCoordinateHexahedron27, GetJacobianGradient) {
  std::srand(31415926);
  for (int i_cell = 1 << 5; i_cell > 0; --i_cell) {
    // build a hexa-gauss and a Lagrange basis on it
    auto a = 20.0, b = 30.0, c = 40.0;
    auto cell = Coordinate {
      // corner nodes on the bottom face
      Global(rand_f() - a, rand_f() - b, rand_f() - c),
      Global(rand_f() + a, rand_f() - b, rand_f() - c),
      Global(rand_f() + a, rand_f() + b, rand_f() - c),
      Global(rand_f() - a, rand_f() + b, rand_f() - c),
      // corner nodes on the top face
      Global(rand_f() - a, rand_f() - b, rand_f() + c),
      Global(rand_f() + a, rand_f() - b, rand_f() + c),
      Global(rand_f() + a, rand_f() + b, rand_f() + c),
      Global(rand_f() - a, rand_f() + b, rand_f() + c),
      // mid-edge nodes on the bottom face
      Global(rand_f(), rand_f() - b, rand_f() - c),
      Global(rand_f() + a, rand_f(), rand_f() - c),
      Global(rand_f(), rand_f() + b, rand_f() - c),
      Global(rand_f() - a, rand_f(), rand_f() - c),
      // mid-edge nodes on vertical edges
      Global(rand_f() - a, rand_f() - b, rand_f()),
      Global(rand_f() + a, rand_f() - b, rand_f()),
      Global(rand_f() + a, rand_f() + b, rand_f()),
      Global(rand_f() - a, rand_f() + b, rand_f()),
      // mid-edge nodes on the top face
      Global(rand_f(), rand_f() - b, rand_f() + c),
      Global(rand_f() + a, rand_f(), rand_f() + c),
      Global(rand_f(), rand_f() + b, rand_f() + c),
      Global(rand_f() - a, rand_f(), rand_f() + c),
      // mid-face nodes
      Global(rand_f(), rand_f(), rand_f() - c),
      Global(rand_f(), rand_f() - b, rand_f()),
      Global(rand_f() + a, rand_f(), rand_f()),
      Global(rand_f(), rand_f() + b, rand_f()),
      Global(rand_f() - a, rand_f(), rand_f()),
      Global(rand_f(), rand_f(), rand_f() + c),
      // center node
      Global(rand_f(), rand_f(), rand_f()),
    };
    // compare gradients with O(h^2) finite difference derivatives
    for (int i_point = 1 << 5; i_point > 0; --i_point) {
      auto x = rand_f(), y = rand_f(), z = rand_f();
      auto local = Local{x, y, z};
      auto mat_grad = cell.LocalToJacobianGradient(local);
      auto det_grad = cell.LocalToJacobianDeterminantGradient(local);
      auto h = 1e-5;
      auto mat_left = cell.LocalToJacobian(x - h, y, z);
      auto mat_right = cell.LocalToJacobian(x + h, y, z);
      mat_grad[X] -= (mat_right - mat_left) / (2 * h);
      auto det_left = mat_left.determinant();
      auto det_right = mat_right.determinant();
      det_grad[X] -= (det_right - det_left) / (2 * h);
      EXPECT_NEAR(mat_grad[X].norm(), 0.0, 1e-8);
      mat_left = cell.LocalToJacobian(x, y - h, z);
      mat_right = cell.LocalToJacobian(x, y + h, z);
      mat_grad[Y] -= (mat_right - mat_left) / (2 * h);
      det_left = mat_left.determinant();
      det_right = mat_right.determinant();
      det_grad[Y] -= (det_right - det_left) / (2 * h);
      EXPECT_NEAR(mat_grad[Y].norm(), 0.0, 1e-8);
      mat_left = cell.LocalToJacobian(x, y, z - h);
      mat_right = cell.LocalToJacobian(x, y, z + h);
      mat_grad[Z] -= (mat_right - mat_left) / (2 * h);
      det_left = mat_left.determinant();
      det_right = mat_right.determinant();
      det_grad[Z] -= (det_right - det_left) / (2 * h);
      EXPECT_NEAR(mat_grad[Z].norm(), 0.0, 1e-8);
      auto det = cell.LocalToJacobian(x, y, z).determinant();
      EXPECT_NEAR(det_grad.norm(), 0.0, det * 1e-9);
    }
    for (int i_point = 1 << 5; i_point > 0; --i_point) {
      auto x = rand_f(), y = rand_f(), z = rand_f();
      auto local = Local{x, y, z};
      auto det = cell.LocalToJacobian(local).determinant();
      auto det_hess = cell.LocalToJacobianDeterminantHessian(local);
      auto h = 1e-5;
      Gradient det_grad_diff = (
          cell.LocalToJacobianDeterminantGradient(Local(x + h, y, z)) -
          cell.LocalToJacobianDeterminantGradient(Local(x - h, y, z))
      ) / (2 * h);
      EXPECT_NEAR(det_hess[XX], det_grad_diff[X], det * 1e-10);
      EXPECT_NEAR(det_hess[XY], det_grad_diff[Y], det * 1e-10);
      EXPECT_NEAR(det_hess[XZ], det_grad_diff[Z], det * 1e-10);
      det_grad_diff = (
          cell.LocalToJacobianDeterminantGradient(Local(x, y + h, z)) -
          cell.LocalToJacobianDeterminantGradient(Local(x, y - h, z))
      ) / (2 * h);
      EXPECT_NEAR(det_hess[YX], det_grad_diff[X], det * 1e-9);
      EXPECT_NEAR(det_hess[YY], det_grad_diff[Y], det * 1e-9);
      EXPECT_NEAR(det_hess[YZ], det_grad_diff[Z], det * 1e-9);
      det_grad_diff = (
          cell.LocalToJacobianDeterminantGradient(Local(x, y, z + h)) -
          cell.LocalToJacobianDeterminantGradient(Local(x, y, z - h))
      ) / (2 * h);
      EXPECT_NEAR(det_hess[ZX], det_grad_diff[X], det * 1e-10);
      EXPECT_NEAR(det_hess[ZY], det_grad_diff[Y], det * 1e-10);
      EXPECT_NEAR(det_hess[ZZ], det_grad_diff[Z], det * 1e-9);
    }
  }
}
TEST_F(TestCoordinateHexahedron27, SortNodesOnFace) {
  using mini::geometry::SortNodesOnFace;
  auto cell = Coordinate{
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
    Coord(0, -10, 0), Coord(+10, 0, 0),
    Coord(0, +10, 0), Coord(-10, 0, 0),
    Coord(0, 0, +10),
    // center node
    Coord(0, 0, 0),
  };
  EXPECT_EQ(cell.GetOutwardNormalVector(0), Coord(0, 0, -1));
  EXPECT_EQ(cell.GetOutwardNormalVector(1), Coord(0, -1, 0));
  EXPECT_EQ(cell.GetOutwardNormalVector(2), Coord(+1, 0, 0));
  EXPECT_EQ(cell.GetOutwardNormalVector(3), Coord(0, +1, 0));
  EXPECT_EQ(cell.GetOutwardNormalVector(4), Coord(-1, 0, 0));
  EXPECT_EQ(cell.GetOutwardNormalVector(5), Coord(0, 0, +1));
  int face_n_node = 9;
  // test the version without conversion:
  {
    using Vector = std::vector<size_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 1010,
      1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020,
      2121, 2222, 2323, 2424, 2525, 2626, 2727, 00 };
    Vector face_nodes, face_nodes_expect;
    face_nodes = { 11, 22, 33, 44, 2121, 1111, 1010, 1212, 99, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 1212, 1111, 1010, 99, 2121, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 22, 55, 66, 2222, 1313, 1414, 1717, 99, 0 };
    face_nodes_expect = { 11, 22, 66, 55, 99, 1414, 1717, 1313, 2222, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 44, 55, 88, 2525, 1212, 1313, 2020, 1616, 0 };
    face_nodes_expect = { 11, 55, 88, 44, 1313, 2020, 1616, 1212, 2525, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 66, 77, 2323, 1818, 1515, 1414, 1010, 0 };
    face_nodes_expect = { 22, 33, 77, 66, 1010, 1515, 1818, 1414, 2323, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 33, 44, 77, 88, 2424, 1919, 1616, 1515, 1111, 0 };
    face_nodes_expect = { 33, 44, 88, 77, 1111, 1616, 1919, 1515, 2424, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 66, 77, 88, 2626, 2020, 1919, 1818, 1717, 0 };
    face_nodes_expect = { 55, 66, 77, 88, 1717, 1818, 1919, 2020, 2626, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
  // test the version with conversion:
  {
    using Vector = std::vector<int16_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 1010,
      1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020,
      2121, 2222, 2323, 2424, 2525, 2626, 2727, 00 };
    Vector face_nodes, face_nodes_expect;
    face_nodes = { 11, 22, 33, 44, 2121, 1111, 1010, 1212, 99, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 1212, 1111, 1010, 99, 2121, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 22, 55, 66, 2222, 1313, 1414, 1717, 99, 0 };
    face_nodes_expect = { 11, 22, 66, 55, 99, 1414, 1717, 1313, 2222, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 44, 55, 88, 2525, 1212, 1313, 2020, 1616, 0 };
    face_nodes_expect = { 11, 55, 88, 44, 1313, 2020, 1616, 1212, 2525, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 66, 77, 2323, 1818, 1515, 1414, 1010, 0 };
    face_nodes_expect = { 22, 33, 77, 66, 1010, 1515, 1818, 1414, 2323, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 33, 44, 77, 88, 2424, 1919, 1616, 1515, 1111, 0 };
    face_nodes_expect = { 33, 44, 88, 77, 1111, 1616, 1919, 1515, 2424, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 66, 77, 88, 2626, 2020, 1919, 1818, 1717, 0 };
    face_nodes_expect = { 55, 66, 77, 88, 1717, 1818, 1919, 2020, 2626, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
}


class TestCoordinateHexahedron26 : public ::testing::Test {
 protected:
  using Coordinate = mini::geometry::Hexahedron26<double>;
  using Coord = typename Coordinate::Global;
  using Global = Coord;
  using Local = Coord;
  using Gradient = Coord;
};
TEST_F(TestCoordinateHexahedron26, CoordinateMap) {
  auto hexa = Coordinate {
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
    Coord(0, -10, 0), Coord(+10, 0, 0),
    Coord(0, +10, 0), Coord(-10, 0, 0),
    Coord(0, 0, +10),
  };
  static_assert(hexa.CellDim() == 3);
  static_assert(hexa.PhysDim() == 3);
  EXPECT_EQ(hexa.CountCorners(), 8);
  EXPECT_EQ(hexa.CountNodes(), 26);
  EXPECT_EQ(hexa.LocalToGlobal(1, 1, 1), Coord(10, 10, 10));
  EXPECT_EQ(hexa.LocalToGlobal(1.5, 1.5, 1.5), Coord(15, 15, 15));
  EXPECT_EQ(hexa.LocalToGlobal(3, 4, 5), Coord(30, 40, 50));
  EXPECT_NEAR(0, (hexa.GlobalToLocal(30, 40, 20) - Coord(3, 4, 2)).norm(),
      1e-10);
  EXPECT_NEAR(0, (hexa.GlobalToLocal(40, 55, 25) - Coord(4, 5.5, 2.5)).norm(),
      1e-10);
  EXPECT_NEAR(0, (hexa.GlobalToLocal(70, 130, 60) - Coord(7, 13, 6)).norm(),
      1e-10);
  mini::geometry::Cell<typename Coordinate::Real> &cell = hexa;
  // test the partition-of-unity property:
  std::srand(31415926);
  for (int i = 0; i < 1000; ++i) {
    auto x = rand_f(), y = rand_f(), z = rand_f();
    auto shapes = cell.LocalToShapeFunctions(x, y, z);
    auto sum = std::accumulate(shapes.begin(), shapes.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-15);
    auto grads = cell.LocalToShapeGradients(x, y, z);
    Coord grads_sum
        = std::accumulate(grads.begin(), grads.end(), Coord(0, 0, 0));
    EXPECT_NEAR(grads_sum.norm(), 0.0, 1e-14);
    // compare gradients with O(h^2) finite difference derivatives
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
}
TEST_F(TestCoordinateHexahedron26, GetJacobianGradient) {
  std::srand(31415926);
  for (int i_cell = 1 << 5; i_cell > 0; --i_cell) {
    // build a hexa-gauss and a Lagrange basis on it
    auto a = 20.0, b = 30.0, c = 40.0;
    auto cell = Coordinate {
      // corner nodes on the bottom face
      Global(rand_f() - a, rand_f() - b, rand_f() - c),
      Global(rand_f() + a, rand_f() - b, rand_f() - c),
      Global(rand_f() + a, rand_f() + b, rand_f() - c),
      Global(rand_f() - a, rand_f() + b, rand_f() - c),
      // corner nodes on the top face
      Global(rand_f() - a, rand_f() - b, rand_f() + c),
      Global(rand_f() + a, rand_f() - b, rand_f() + c),
      Global(rand_f() + a, rand_f() + b, rand_f() + c),
      Global(rand_f() - a, rand_f() + b, rand_f() + c),
      // mid-edge nodes on the bottom face
      Global(rand_f(), rand_f() - b, rand_f() - c),
      Global(rand_f() + a, rand_f(), rand_f() - c),
      Global(rand_f(), rand_f() + b, rand_f() - c),
      Global(rand_f() - a, rand_f(), rand_f() - c),
      // mid-edge nodes on vertical edges
      Global(rand_f() - a, rand_f() - b, rand_f()),
      Global(rand_f() + a, rand_f() - b, rand_f()),
      Global(rand_f() + a, rand_f() + b, rand_f()),
      Global(rand_f() - a, rand_f() + b, rand_f()),
      // mid-edge nodes on the top face
      Global(rand_f(), rand_f() - b, rand_f() + c),
      Global(rand_f() + a, rand_f(), rand_f() + c),
      Global(rand_f(), rand_f() + b, rand_f() + c),
      Global(rand_f() - a, rand_f(), rand_f() + c),
      // mid-face nodes
      Global(rand_f(), rand_f(), rand_f() - c),
      Global(rand_f(), rand_f() - b, rand_f()),
      Global(rand_f() + a, rand_f(), rand_f()),
      Global(rand_f(), rand_f() + b, rand_f()),
      Global(rand_f() - a, rand_f(), rand_f()),
      Global(rand_f(), rand_f(), rand_f() + c),
    };
    // compare gradients with O(h^2) finite difference derivatives
    for (int i_point = 1 << 5; i_point > 0; --i_point) {
      auto x = rand_f(), y = rand_f(), z = rand_f();
      auto local = Local{x, y, z};
      auto mat_grad = cell.LocalToJacobianGradient(local);
      auto det_grad = cell.LocalToJacobianDeterminantGradient(local);
      auto h = 1e-5;
      auto mat_left = cell.LocalToJacobian(x - h, y, z);
      auto mat_right = cell.LocalToJacobian(x + h, y, z);
      mat_grad[X] -= (mat_right - mat_left) / (2 * h);
      auto det_left = mat_left.determinant();
      auto det_right = mat_right.determinant();
      det_grad[X] -= (det_right - det_left) / (2 * h);
      EXPECT_NEAR(mat_grad[X].norm(), 0.0, 1e-8);
      mat_left = cell.LocalToJacobian(x, y - h, z);
      mat_right = cell.LocalToJacobian(x, y + h, z);
      mat_grad[Y] -= (mat_right - mat_left) / (2 * h);
      det_left = mat_left.determinant();
      det_right = mat_right.determinant();
      det_grad[Y] -= (det_right - det_left) / (2 * h);
      EXPECT_NEAR(mat_grad[Y].norm(), 0.0, 1e-8);
      mat_left = cell.LocalToJacobian(x, y, z - h);
      mat_right = cell.LocalToJacobian(x, y, z + h);
      mat_grad[Z] -= (mat_right - mat_left) / (2 * h);
      det_left = mat_left.determinant();
      det_right = mat_right.determinant();
      det_grad[Z] -= (det_right - det_left) / (2 * h);
      EXPECT_NEAR(mat_grad[Z].norm(), 0.0, 1e-8);
      auto det = cell.LocalToJacobian(x, y, z).determinant();
      EXPECT_NEAR(det_grad.norm(), 0.0, det * 1e-9);
    }
    for (int i_point = 1 << 5; i_point > 0; --i_point) {
      auto x = rand_f(), y = rand_f(), z = rand_f();
      auto local = Local{x, y, z};
      auto det = cell.LocalToJacobian(local).determinant();
      auto det_hess = cell.LocalToJacobianDeterminantHessian(local);
      auto h = 1e-5;
      Gradient det_grad_diff = (
          cell.LocalToJacobianDeterminantGradient(Local(x + h, y, z)) -
          cell.LocalToJacobianDeterminantGradient(Local(x - h, y, z))
      ) / (2 * h);
      EXPECT_NEAR(det_hess[XX], det_grad_diff[X], det * 1e-9);
      EXPECT_NEAR(det_hess[XY], det_grad_diff[Y], det * 1e-10);
      EXPECT_NEAR(det_hess[XZ], det_grad_diff[Z], det * 1e-10);
      det_grad_diff = (
          cell.LocalToJacobianDeterminantGradient(Local(x, y + h, z)) -
          cell.LocalToJacobianDeterminantGradient(Local(x, y - h, z))
      ) / (2 * h);
      EXPECT_NEAR(det_hess[YX], det_grad_diff[X], det * 1e-10);
      EXPECT_NEAR(det_hess[YY], det_grad_diff[Y], det * 1e-9);
      EXPECT_NEAR(det_hess[YZ], det_grad_diff[Z], det * 1e-9);
      det_grad_diff = (
          cell.LocalToJacobianDeterminantGradient(Local(x, y, z + h)) -
          cell.LocalToJacobianDeterminantGradient(Local(x, y, z - h))
      ) / (2 * h);
      EXPECT_NEAR(det_hess[ZX], det_grad_diff[X], det * 1e-10);
      EXPECT_NEAR(det_hess[ZY], det_grad_diff[Y], det * 1e-9);
      EXPECT_NEAR(det_hess[ZZ], det_grad_diff[Z], det * 1e-10);
    }
  }
}
TEST_F(TestCoordinateHexahedron26, SortNodesOnFace) {
  using mini::geometry::SortNodesOnFace;
  auto cell = Coordinate{
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
    Coord(0, -10, 0), Coord(+10, 0, 0),
    Coord(0, +10, 0), Coord(-10, 0, 0),
    Coord(0, 0, +10),
  };
  EXPECT_EQ(cell.GetOutwardNormalVector(0), Coord(0, 0, -1));
  EXPECT_EQ(cell.GetOutwardNormalVector(1), Coord(0, -1, 0));
  EXPECT_EQ(cell.GetOutwardNormalVector(2), Coord(+1, 0, 0));
  EXPECT_EQ(cell.GetOutwardNormalVector(3), Coord(0, +1, 0));
  EXPECT_EQ(cell.GetOutwardNormalVector(4), Coord(-1, 0, 0));
  EXPECT_EQ(cell.GetOutwardNormalVector(5), Coord(0, 0, +1));
  int face_n_node = 9;
  // test the version without conversion:
  {
    using Vector = std::vector<size_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 1010,
      1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020,
      2121, 2222, 2323, 2424, 2525, 2626, 00 };
    Vector face_nodes, face_nodes_expect;
    face_nodes = { 11, 22, 33, 44, 2121, 1111, 1010, 1212, 99, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 1212, 1111, 1010, 99, 2121, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 22, 55, 66, 2222, 1313, 1414, 1717, 99, 0 };
    face_nodes_expect = { 11, 22, 66, 55, 99, 1414, 1717, 1313, 2222, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 44, 55, 88, 2525, 1212, 1313, 2020, 1616, 0 };
    face_nodes_expect = { 11, 55, 88, 44, 1313, 2020, 1616, 1212, 2525, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 66, 77, 2323, 1818, 1515, 1414, 1010, 0 };
    face_nodes_expect = { 22, 33, 77, 66, 1010, 1515, 1818, 1414, 2323, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 33, 44, 77, 88, 2424, 1919, 1616, 1515, 1111, 0 };
    face_nodes_expect = { 33, 44, 88, 77, 1111, 1616, 1919, 1515, 2424, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 66, 77, 88, 2626, 2020, 1919, 1818, 1717, 0 };
    face_nodes_expect = { 55, 66, 77, 88, 1717, 1818, 1919, 2020, 2626, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
  // test the version with conversion:
  {
    using Vector = std::vector<int16_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 1010,
      1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020,
      2121, 2222, 2323, 2424, 2525, 2626, 00 };
    Vector face_nodes, face_nodes_expect;
    face_nodes = { 11, 22, 33, 44, 2121, 1111, 1010, 1212, 99, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 1212, 1111, 1010, 99, 2121, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 22, 55, 66, 2222, 1313, 1414, 1717, 99, 0 };
    face_nodes_expect = { 11, 22, 66, 55, 99, 1414, 1717, 1313, 2222, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 44, 55, 88, 2525, 1212, 1313, 2020, 1616, 0 };
    face_nodes_expect = { 11, 55, 88, 44, 1313, 2020, 1616, 1212, 2525, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 66, 77, 2323, 1818, 1515, 1414, 1010, 0 };
    face_nodes_expect = { 22, 33, 77, 66, 1010, 1515, 1818, 1414, 2323, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 33, 44, 77, 88, 2424, 1919, 1616, 1515, 1111, 0 };
    face_nodes_expect = { 33, 44, 88, 77, 1111, 1616, 1919, 1515, 2424, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 66, 77, 88, 2626, 2020, 1919, 1818, 1717, 0 };
    face_nodes_expect = { 55, 66, 77, 88, 1717, 1818, 1919, 2020, 2626, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
