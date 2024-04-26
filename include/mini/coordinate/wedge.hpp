//  Copyright 2023 PEI Weicheng
#ifndef MINI_GEOMETRY_WEDGE_HPP_
#define MINI_GEOMETRY_WEDGE_HPP_

#include <concepts>

#include <cassert>

#include <array>
#include <initializer_list>
#include <vector>

#include "mini/geometry/element.hpp"
#include "mini/geometry/cell.hpp"

namespace mini {
namespace geometry {

/**
 * @brief Abstract coordinate map on wedge elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Wedge : public Cell<Scalar> {
  using Base = Cell<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  int CountCorners() const final {
    return 6;
  }
  const Global &center() const final {
    return center_;
  }

 protected:
  Global center_;
  void _BuildCenter() final {
    Scalar a = 1.0 / 3;
    center_ = this->LocalToGlobal(a, a, 0);
  }

  static int GetTriangleId(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node/* number of nodes on triangle */) {
    int i_face = 0;
    for (int f = 0; f < face_n_node; ++f) {
      if (cell_nodes[4] == face_nodes[f]) {
        i_face = 1;
        break;
      }
    }
    return i_face;
  }

  static int GetQuadrangleId(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node/* number of nodes on quadragle */) {
    int c_sum = 0;  // sum of c's in (0, 1, 2)
    for (int f = 0; f < face_n_node; ++f) {
      for (int c = 0; c < 3; ++c) {
        if (cell_nodes[c] == face_nodes[f]) {
          c_sum += c;
          break;
        }
      }
    }
    int i_face;
    switch (c_sum) {
    case 1: i_face = 0; break;
    case 2: i_face = 2; break;
    case 3: i_face = 1; break;
    default: assert(false);
    }
    return i_face;
  }
};

/**
 * @brief Coordinate map on 6-node wedge elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Wedge6 : public Wedge<Scalar> {
  using Base = Wedge<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  static constexpr int kNodes = 6;

 private:
  std::array<Global, kNodes> global_coords_;
  static const std::array<Local, kNodes> local_coords_;
  static const std::array<std::array<int, 3>, 2> triangles_;
  static const std::array<std::array<int, 4>, 3> quadrangles_;

 public:
  int CountNodes() const final {
    return kNodes;
  }
  void SortNodesOnFace(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node) const final {
    if (face_n_node == 3) {
      int i_face = Base::GetTriangleId(cell_nodes, face_nodes, face_n_node);
      for (int f = 0; f < face_n_node; ++f) {
        face_nodes[f] = cell_nodes[triangles_[i_face][f]];
      }
    } else {
      assert(face_n_node == 4);
      int i_face = Base::GetQuadrangleId(cell_nodes, face_nodes, face_n_node);
      for (int f = 0; f < face_n_node; ++f) {
        face_nodes[f] = cell_nodes[quadrangles_[i_face][f]];
      }
    }
  }

 public:
  static void LocalToShapeFunctions(Scalar a_local, Scalar b_local,
      Scalar z_local, Scalar *shapes) {
    auto c_local = 1.0 - a_local - b_local;
    auto factor_z = (1 - z_local) / 2;
    shapes[0] = a_local * factor_z;
    shapes[1] = b_local * factor_z;
    shapes[2] = c_local * factor_z;
    factor_z += z_local;
    shapes[3] = a_local * factor_z;
    shapes[4] = b_local * factor_z;
    shapes[5] = c_local * factor_z;
  }
  std::vector<Scalar> LocalToShapeFunctions(
      Scalar a_local, Scalar b_local, Scalar z_local) const final {
    auto shapes = std::vector<Scalar>(kNodes);
    LocalToShapeFunctions(a_local, b_local, z_local, shapes.data());
    return shapes;
  }
  static void LocalToShapeGradients(Scalar a_local, Scalar b_local,
      Scalar z_local, Local *grads) {
    auto c_local = 1.0 - a_local - b_local;
    auto factor_z = (1 - z_local) / 2;
    // shapes[0] = a_local * factor_z
    grads[0][A] = factor_z;
    grads[0][B] = 0;
    grads[0][Z] = -0.5 * a_local;
    // shapes[1] = b_local * factor_z
    grads[1][A] = 0;
    grads[1][B] = factor_z;
    grads[1][Z] = -0.5 * b_local;
    // shapes[2] = (1 - a_local - b_local) * factor_z
    grads[2][A] = -factor_z;
    grads[2][B] = -factor_z;
    grads[2][Z] = -0.5 * c_local;
    factor_z += z_local;
    // shapes[3] = a_local * factor_z;
    grads[3][A] = factor_z;
    grads[3][B] = 0;
    grads[3][Z] = +0.5 * a_local;
    // shapes[4] = b_local * factor_z;
    grads[4][A] = 0;
    grads[4][B] = factor_z;
    grads[4][Z] = +0.5 * b_local;
    // shapes[5] = c_local * factor_z;
    grads[5][A] = -factor_z;
    grads[5][B] = -factor_z;
    grads[5][Z] = +0.5 * c_local;
  }
  std::vector<Local> LocalToShapeGradients(
      Scalar a_local, Scalar b_local, Scalar z_local) const final {
    auto grads = std::vector<Local>(kNodes);
    LocalToShapeGradients(a_local, b_local, z_local, grads.data());
    return grads;
  }

 public:
  Global const &GetGlobalCoord(int i) const final {
    assert(0 <= i && i < CountNodes());
    return global_coords_[i];
  }
  Local const &GetLocalCoord(int i) const final {
    assert(0 <= i && i < CountNodes());
    return local_coords_[i];
  }

 public:
  Wedge6(std::initializer_list<Global> il) {
    Element<Scalar, 3, 3>::_Build(this, il);
  }
};
// initialization of static const members:
template <std::floating_point Scalar>
const std::array<typename Wedge6<Scalar>::Local, 6>
Wedge6<Scalar>::local_coords_{
  Wedge6::Local(1, 0, -1), Wedge6::Local(0, 1, -1), Wedge6::Local(0, 0, -1),
  Wedge6::Local(1, 0, +1), Wedge6::Local(0, 1, +1), Wedge6::Local(0, 0, +1)
};
/* See http://cgns.github.io/CGNS_docs_current/sids/conv.figs/penta_6.png for node numbering.
*/
template <std::floating_point Scalar>
const std::array<std::array<int, 3>, 2>
Wedge6<Scalar>::triangles_{
  0, 2, 1,
  3, 4, 5,
};
template <std::floating_point Scalar>
const std::array<std::array<int, 4>, 3>
Wedge6<Scalar>::quadrangles_{
  0, 1, 4, 3,
  1, 2, 5, 4,
  0, 3, 5, 2,
};

template <std::floating_point Scalar>
class Wedge18;

/**
 * @brief Coordinate map on 15-node wedge elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Wedge15 : public Wedge<Scalar> {
  using Base = Wedge<Scalar>;
  friend class Wedge18<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  static constexpr int kNodes = 15;

 private:
  std::array<Global, kNodes> global_coords_;
  static const std::array<Local, kNodes> local_coords_;
  static const std::array<std::array<int, 6>, 2> triangles_;
  static const std::array<std::array<int, 8>, 3> quadrangles_;

 public:
  int CountNodes() const final {
    return kNodes;
  }
  void SortNodesOnFace(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node) const final {
    if (face_n_node == 6) {
      int i_face = Base::GetTriangleId(cell_nodes, face_nodes, face_n_node);
      for (int f = 0; f < face_n_node; ++f) {
        face_nodes[f] = cell_nodes[triangles_[i_face][f]];
      }
    } else {
      assert(face_n_node == 8);
      int i_face = Base::GetQuadrangleId(cell_nodes, face_nodes, face_n_node);
      for (int f = 0; f < face_n_node; ++f) {
        face_nodes[f] = cell_nodes[quadrangles_[i_face][f]];
      }
    }
  }

 private:
  static void LocalToNewShapeFunctions(Scalar a_local, Scalar b_local,
      Scalar z_local, Scalar *shapes) {
    auto c_local = 1.0 - a_local - b_local;
    auto factor_ab = a_local * b_local * 4;
    auto factor_bc = b_local * c_local * 4;
    auto factor_ca = c_local * a_local * 4;
    // local_coords_[i][Z] = -1
    auto factor_z = (1 - z_local) / 2;
    shapes[6] = factor_ab * factor_z;
    shapes[7] = factor_bc * factor_z;
    shapes[8] = factor_ca * factor_z;
    // local_coords_[i][Z] = 0
    factor_z = 1 - z_local * z_local;
    shapes[9] = a_local * factor_z;
    shapes[10] = b_local * factor_z;
    shapes[11] = c_local * factor_z;
    // local_coords_[i][Z] = +1
    factor_z = (1 + z_local) / 2;
    shapes[12] = factor_ab * factor_z;
    shapes[13] = factor_bc * factor_z;
    shapes[14] = factor_ca * factor_z;
  }
  static void LocalToNewShapeGradients(Scalar a_local, Scalar b_local,
      Scalar z_local, Local *grads) {
    auto c_local = 1.0 - a_local - b_local;
    auto factor_a = 4 * a_local;
    auto factor_b = 4 * b_local;
    auto factor_c = 4 * c_local;
    auto factor_ab = factor_a * b_local;
    auto factor_bc = factor_b * c_local;
    auto factor_ca = factor_c * a_local;
    // local_coords_[i][Z] = -1
    auto factor_z = (1 - z_local) / 2;
    auto grad_z = -0.5;
    // shape[6] = 4 * a * b * factor_z
    grads[6][A] = factor_b * factor_z;
    grads[6][B] = factor_a * factor_z;
    grads[6][Z] = factor_ab * grad_z;
    // shape[7] = 4 * b * c * factor_z
    grads[7][A] = -factor_b * factor_z;
    grads[7][B] = (factor_c - factor_b) * factor_z;
    grads[7][Z] = factor_bc * grad_z;
    // shape[8] = 4 * c * a * factor_z
    grads[8][A] = (factor_c - factor_a) * factor_z;
    grads[8][B] = -factor_a * factor_z;
    grads[8][Z] = factor_ca * grad_z;
    // local_coords_[i][Z] = 0
    factor_z = (1 - z_local * z_local);
    grad_z = -2 * z_local;
    // shape[9] = a_local * factor_z;
    grads[9][A] = factor_z;
    grads[9][B] = 0;
    grads[9][Z] = a_local * grad_z;
    // shape[10] = b_local * factor_z;
    grads[10][A] = 0;
    grads[10][B] = factor_z;
    grads[10][Z] = b_local * grad_z;
    // shape[11] = c_local * factor_z;
    grads[11][A] = -factor_z;
    grads[11][B] = -factor_z;
    grads[11][Z] = c_local * grad_z;
    // local_coords_[i][Z] = +1
    factor_z = (1 + z_local) / 2;
    grad_z = +0.5;
    // shape[12] = 4 * a * b * factor_z
    grads[12][A] = factor_b * factor_z;
    grads[12][B] = factor_a * factor_z;
    grads[12][Z] = factor_ab * grad_z;
    // shape[13] = 4 * b * c * factor_z
    grads[13][A] = -factor_b * factor_z;
    grads[13][B] = (factor_c - factor_b) * factor_z;
    grads[13][Z] = factor_bc * grad_z;
    // shape[14] = 4 * c * a * factor_z
    grads[14][A] = (factor_c - factor_a) * factor_z;
    grads[14][B] = -factor_a * factor_z;
    grads[14][Z] = factor_ca * grad_z;
  }

 public:
  static void LocalToShapeFunctions(Scalar a_local, Scalar b_local,
      Scalar z_local, Scalar *shapes) {
    Wedge6<Scalar>::LocalToShapeFunctions(a_local, b_local, z_local, shapes);
    LocalToNewShapeFunctions(a_local, b_local, z_local, shapes);
    for (int j = 6; j < 15; ++j) {
      Scalar a_j = local_coords_[j][A];
      Scalar b_j = local_coords_[j][B];
      Scalar z_j = local_coords_[j][Z];
      Scalar old_shapes_on_new_nodes[6];
      Wedge6<Scalar>::LocalToShapeFunctions(
          a_j, b_j, z_j, old_shapes_on_new_nodes);
      for (int i = 0; i < 6; ++i) {
        shapes[i] -= old_shapes_on_new_nodes[i] * shapes[j];
      }
    }
  }
  std::vector<Scalar> LocalToShapeFunctions(
      Scalar a_local, Scalar b_local, Scalar z_local) const final {
    auto shapes = std::vector<Scalar>(kNodes);
    LocalToShapeFunctions(a_local, b_local, z_local, shapes.data());
    return shapes;
  }
  static void LocalToShapeGradients(Scalar a_local, Scalar b_local,
      Scalar z_local, Local *grads) {
    Wedge6<Scalar>::LocalToShapeGradients(a_local, b_local, z_local, grads);
    LocalToNewShapeGradients(a_local, b_local, z_local, grads);
    for (int j = 6; j < 15; ++j) {
      Scalar a_j = local_coords_[j][A];
      Scalar b_j = local_coords_[j][B];
      Scalar z_j = local_coords_[j][Z];
      Scalar old_shapes_on_new_nodes[6];
      Wedge6<Scalar>::LocalToShapeFunctions(
          a_j, b_j, z_j, old_shapes_on_new_nodes);
      for (int i = 0; i < 6; ++i) {
        grads[i] -= old_shapes_on_new_nodes[i] * grads[j];
      }
    }
  }
  std::vector<Local> LocalToShapeGradients(
      Scalar a_local, Scalar b_local, Scalar z_local) const final {
    auto grads = std::vector<Local>(kNodes);
    LocalToShapeGradients(a_local, b_local, z_local, grads.data());
    return grads;
  }

 public:
  Global const &GetGlobalCoord(int i) const final {
    assert(0 <= i && i < CountNodes());
    return global_coords_[i];
  }
  Local const &GetLocalCoord(int i) const final {
    assert(0 <= i && i < CountNodes());
    return local_coords_[i];
  }

 public:
  Wedge15(std::initializer_list<Global> il) {
    Element<Scalar, 3, 3>::_Build(this, il);
  }
};
// initialization of static const members:
template <std::floating_point Scalar>
const std::array<typename Wedge15<Scalar>::Local, 15>
Wedge15<Scalar>::local_coords_{
  // corner nodes on the bottom face
  Wedge15::Local(1, 0, -1), Wedge15::Local(0, 1, -1), Wedge15::Local(0, 0, -1),
  // corner nodes on the top face
  Wedge15::Local(1, 0, +1), Wedge15::Local(0, 1, +1), Wedge15::Local(0, 0, +1),
  // mid-edge nodes on the bottom face
  Wedge15::Local(0.5, 0.5, -1), Wedge15::Local(0, 0.5, -1),
  Wedge15::Local(0.5, 0, -1),
  // mid-edge nodes on vertical edges
  Wedge15::Local(1, 0, 0), Wedge15::Local(0, 1, 0), Wedge15::Local(0, 0, 0),
  // mid-edge nodes on the top face
  Wedge15::Local(0.5, 0.5, +1), Wedge15::Local(0, 0.5, +1),
  Wedge15::Local(0.5, 0, +1),
};
/* See http://cgns.github.io/CGNS_docs_current/sids/conv.figs/penta_15.png for node numbering.
*/
template <std::floating_point Scalar>
const std::array<std::array<int, 6>, 2>
Wedge15<Scalar>::triangles_{
  0, 2, 1, 8, 7, 6,
  3, 4, 5, 12, 13, 14,
};
template <std::floating_point Scalar>
const std::array<std::array<int, 8>, 3>
Wedge15<Scalar>::quadrangles_{
  0, 1, 4, 3, 6, 10, 12, 9,
  1, 2, 5, 4, 7, 11, 13, 10,
  0, 3, 5, 2, 9, 14, 11, 8,
};

/**
 * @brief Coordinate map on 18-node wedge elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Wedge18 : public Wedge<Scalar> {
  using Base = Wedge<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  static constexpr int kNodes = 18;

 private:
  std::array<Global, kNodes> global_coords_;
  static const std::array<Local, kNodes> local_coords_;
  static const std::array<std::array<int, 9>, 3> quadrangles_;

 public:
  int CountNodes() const final {
    return kNodes;
  }
  void SortNodesOnFace(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node) const final {
    if (face_n_node == 6) {
      int i_face = Base::GetTriangleId(cell_nodes, face_nodes, face_n_node);
      for (int f = 0; f < face_n_node; ++f) {
        face_nodes[f] = cell_nodes[Wedge15<Scalar>::triangles_[i_face][f]];
      }
    } else {
      assert(face_n_node == 9);
      int i_face = Base::GetQuadrangleId(cell_nodes, face_nodes, face_n_node);
      for (int f = 0; f < face_n_node; ++f) {
        face_nodes[f] = cell_nodes[quadrangles_[i_face][f]];
      }
    }
  }

 public:
  static void LocalToShapeFunctions(Scalar a_local, Scalar b_local,
      Scalar z_local, Scalar *shapes) {
    auto c_local = 1.0 - a_local - b_local;
    auto quadratic_a = a_local * (2 * a_local - 1);
    auto quadratic_b = b_local * (2 * b_local - 1);
    auto quadratic_c = c_local * (2 * c_local - 1);
    auto factor_ab = a_local * b_local * 4;
    auto factor_bc = b_local * c_local * 4;
    auto factor_ca = c_local * a_local * 4;
    // local_coords_[i][Z] = -1
    auto factor_z = (z_local - 1) * z_local / 2;
    shapes[0] = quadratic_a * factor_z;
    shapes[1] = quadratic_b * factor_z;
    shapes[2] = quadratic_c * factor_z;
    shapes[6] = factor_ab * factor_z;
    shapes[7] = factor_bc * factor_z;
    shapes[8] = factor_ca * factor_z;
    // local_coords_[i][Z] = +1
    factor_z += z_local;  // == (z_local + 1) * z_local / 2
    shapes[3] = quadratic_a * factor_z;
    shapes[4] = quadratic_b * factor_z;
    shapes[5] = quadratic_c * factor_z;
    shapes[12] = factor_ab * factor_z;
    shapes[13] = factor_bc * factor_z;
    shapes[14] = factor_ca * factor_z;
    // local_coords_[i][Z] = 0
    factor_z = 1 - z_local * z_local;
    shapes[9] = quadratic_a * factor_z;
    shapes[10] = quadratic_b * factor_z;
    shapes[11] = quadratic_c * factor_z;
    shapes[15] = factor_ab * factor_z;
    shapes[16] = factor_bc * factor_z;
    shapes[17] = factor_ca * factor_z;
  }
  std::vector<Scalar> LocalToShapeFunctions(
      Scalar a_local, Scalar b_local, Scalar z_local) const final {
    auto shapes = std::vector<Scalar>(kNodes);
    LocalToShapeFunctions(a_local, b_local, z_local, shapes.data());
    return shapes;
  }
  static void LocalToShapeGradients(Scalar a_local, Scalar b_local,
      Scalar z_local, Local *grads) {
    auto c_local = 1.0 - a_local - b_local;
    auto quadratic_a = a_local * (2 * a_local - 1);
    auto quadratic_b = b_local * (2 * b_local - 1);
    auto quadratic_c = c_local * (2 * c_local - 1);
    auto factor_a = a_local * 4;
    auto factor_b = b_local * 4;
    auto factor_c = c_local * 4;
    auto factor_ab = factor_a * b_local;
    auto factor_bc = factor_b * c_local;
    auto factor_ca = factor_c * a_local;
    // local_coords_[i][Z] = -1
    auto factor_z = (z_local - 1) * z_local / 2;
    auto grad_z = z_local - 0.5;
    // shapes[0] = quadratic_a * factor_z;
    grads[0][A] = (factor_a - 1) * factor_z;
    grads[0][B] = 0;
    grads[0][Z] = quadratic_a * grad_z;
    // shapes[1] = quadratic_b * factor_z;
    grads[1][A] = 0;
    grads[1][B] = (factor_b - 1) * factor_z;
    grads[1][Z] = quadratic_b * grad_z;
    // shapes[2] = quadratic_c * factor_z;
    grads[2][A] =
    grads[2][B] = (1 - factor_c) * factor_z;
    grads[2][Z] = quadratic_c * grad_z;
    // shapes[6] = factor_ab * factor_z;
    grads[6][A] = factor_b * factor_z;
    grads[6][B] = factor_a * factor_z;
    grads[6][Z] = factor_ab * grad_z;
    // shapes[7] = factor_bc * factor_z;
    grads[7][A] = -factor_b * factor_z;
    grads[7][B] = (factor_c - factor_b) * factor_z;
    grads[7][Z] = factor_bc * grad_z;
    // shapes[8] = factor_ca * factor_z;
    grads[8][A] = (factor_c - factor_a) * factor_z;
    grads[8][B] = -factor_a * factor_z;
    grads[8][Z] = factor_ca * grad_z;
    // local_coords_[i][Z] = +1
    factor_z += z_local;  // == (z_local + 1) * z_local / 2
    grad_z += 1;
    // shapes[3] = quadratic_a * factor_z;
    grads[3][A] = (factor_a - 1) * factor_z;
    grads[3][B] = 0;
    grads[3][Z] = quadratic_a * grad_z;
    // shapes[4] = quadratic_b * factor_z;
    grads[4][A] = 0;
    grads[4][B] = (factor_b - 1) * factor_z;
    grads[4][Z] = quadratic_b * grad_z;
    // shapes[5] = quadratic_c * factor_z;
    grads[5][A] =
    grads[5][B] = (1 - factor_c) * factor_z;
    grads[5][Z] = quadratic_c * grad_z;
    // shapes[12] = factor_ab * factor_z;
    grads[12][A] = factor_b * factor_z;
    grads[12][B] = factor_a * factor_z;
    grads[12][Z] = factor_ab * grad_z;
    // shapes[13] = factor_bc * factor_z;
    grads[13][A] = -factor_b * factor_z;
    grads[13][B] = (factor_c - factor_b) * factor_z;
    grads[13][Z] = factor_bc * grad_z;
    // shapes[14] = factor_ca * factor_z;
    grads[14][A] = (factor_c - factor_a) * factor_z;
    grads[14][B] = -factor_a * factor_z;
    grads[14][Z] = factor_ca * grad_z;
    // local_coords_[i][Z] = 0
    factor_z = 1 - z_local * z_local;
    grad_z = -2 * z_local;
    // shapes[9] = quadratic_a * factor_z;
    grads[9][A] = (factor_a - 1) * factor_z;
    grads[9][B] = 0;
    grads[9][Z] = quadratic_a * grad_z;
    // shapes[10] = quadratic_b * factor_z;
    grads[10][A] = 0;
    grads[10][B] = (factor_b - 1) * factor_z;
    grads[10][Z] = quadratic_b * grad_z;
    // shapes[11] = quadratic_c * factor_z;
    grads[11][A] =
    grads[11][B] = (1 - factor_c) * factor_z;
    grads[11][Z] = quadratic_c * grad_z;
    // shapes[15] = factor_ab * factor_z;
    grads[15][A] = factor_b * factor_z;
    grads[15][B] = factor_a * factor_z;
    grads[15][Z] = factor_ab * grad_z;
    // shapes[16] = factor_bc * factor_z;
    grads[16][A] = -factor_b * factor_z;
    grads[16][B] = (factor_c - factor_b) * factor_z;
    grads[16][Z] = factor_bc * grad_z;
    // shapes[17] = factor_ca * factor_z;
    grads[17][A] = (factor_c - factor_a) * factor_z;
    grads[17][B] = -factor_a * factor_z;
    grads[17][Z] = factor_ca * grad_z;
  }
  std::vector<Local> LocalToShapeGradients(
      Scalar a_local, Scalar b_local, Scalar z_local) const final {
    auto grads = std::vector<Local>(kNodes);
    LocalToShapeGradients(a_local, b_local, z_local, grads.data());
    return grads;
  }

 public:
  Global const &GetGlobalCoord(int i) const final {
    assert(0 <= i && i < CountNodes());
    return global_coords_[i];
  }
  Local const &GetLocalCoord(int i) const final {
    assert(0 <= i && i < CountNodes());
    return local_coords_[i];
  }

 public:
  Wedge18(std::initializer_list<Global> il) {
    Element<Scalar, 3, 3>::_Build(this, il);
  }
};
// initialization of static const members:
template <std::floating_point Scalar>
const std::array<typename Wedge18<Scalar>::Local, 18>
Wedge18<Scalar>::local_coords_{
  // corner nodes on the bottom face
  Wedge18::Local(1, 0, -1), Wedge18::Local(0, 1, -1), Wedge18::Local(0, 0, -1),
  // corner nodes on the top face
  Wedge18::Local(1, 0, +1), Wedge18::Local(0, 1, +1), Wedge18::Local(0, 0, +1),
  // mid-edge nodes on the bottom face
  Wedge18::Local(0.5, 0.5, -1), Wedge18::Local(0, 0.5, -1),
  Wedge18::Local(0.5, 0, -1),
  // mid-edge nodes on vertical edges
  Wedge18::Local(1, 0, 0), Wedge18::Local(0, 1, 0), Wedge18::Local(0, 0, 0),
  // mid-edge nodes on the top face
  Wedge18::Local(0.5, 0.5, +1), Wedge18::Local(0, 0.5, +1),
  Wedge18::Local(0.5, 0, +1),
  // mid-face nodes on quadragles
  Wedge18::Local(0.5, 0.5, 0), Wedge18::Local(0, 0.5, 0),
  Wedge18::Local(0.5, 0, 0),
};
/* See http://cgns.github.io/CGNS_docs_current/sids/conv.figs/penta_18.png for node numbering.
 * Only quadrangles are listed here, since triangles are the same as Wedge15.
 */
template <std::floating_point Scalar>
const std::array<std::array<int, 9>, 3>
Wedge18<Scalar>::quadrangles_{
  0, 1, 4, 3, 6, 10, 12, 9, 15,
  1, 2, 5, 4, 7, 11, 13, 10, 16,
  0, 3, 5, 2, 9, 14, 11, 8, 17,
};

}  // namespace geometry
}  // namespace mini

#endif  // MINI_GEOMETRY_WEDGE_HPP_
