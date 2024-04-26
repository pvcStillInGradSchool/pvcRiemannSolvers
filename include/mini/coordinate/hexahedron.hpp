//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_GEOMETRY_HEXAHEDRON_HPP_
#define MINI_GEOMETRY_HEXAHEDRON_HPP_

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
 * @brief Abstract coordinate map on hexahedral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Hexahedron : public Cell<Scalar> {
  using Base = Cell<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;
  using typename Base::Hessian;

  int CountCorners() const final {
    return 8;
  }
  const Global &center() const final {
    return center_;
  }
  Global GetOutwardNormalVector(int i_face) const {
    Jacobian jacobian;
    Global normal;
    switch (i_face) {
    case 0:
      jacobian = this->LocalToJacobian(0, 0, -1);
      normal = -jacobian.row(Z).normalized();
      break;
    case 1:
      jacobian = this->LocalToJacobian(0, -1, 0);
      normal = -jacobian.row(Y).normalized();
      break;
    case 2:
      jacobian = this->LocalToJacobian(+1, 0, 0);
      normal = jacobian.row(X).normalized();
      break;
    case 3:
      jacobian = this->LocalToJacobian(0, +1, 0);
      normal = jacobian.row(Y).normalized();
      break;
    case 4:
      jacobian = this->LocalToJacobian(-1, 0, 0);
      normal = -jacobian.row(X).normalized();
      break;
    case 5:
      jacobian = this->LocalToJacobian(0, 0, +1);
      normal = jacobian.row(Z).normalized();
      break;
    default:
      assert(false);
    }
    return normal;
  }

 protected:
  Global center_;
  void _BuildCenter() final {
    Scalar a = 0;
    center_ = this->LocalToGlobal(a, a, a);
  }

  static int GetFaceId(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node/* number of nodes on quadrangle */) {
    int cnt = 0, c = 0, c_sum = 0;
    while (cnt < 3) {
      auto curr_node = cell_nodes[c];
      for (int f = 0; f < face_n_node; ++f) {
        if (face_nodes[f] == curr_node) {
          c_sum += c;
          ++cnt;
          break;
        }
      }
      ++c;
    }
    int i_face;
    switch (c_sum) {
    case 3:
      i_face = 0; break;
    case 5:
      i_face = 1; break;
    case 7:
      i_face = 4; break;
    case 8:
      i_face = 2; break;
    case 11:
      i_face = 3; break;
    case 15:
      i_face = 5; break;
    default:
      assert(false);
    }
    return i_face;
  }
};

template <std::floating_point Scalar>
class Hexahedron8;

template <std::floating_point Scalar>
class Hexahedron20;

template <std::floating_point Scalar>
class Hexahedron26;

/**
 * @brief Coordinate map on 27-node hexahedral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Hexahedron27 : public Hexahedron<Scalar> {
  using Base = Hexahedron<Scalar>;
  friend class Hexahedron8<Scalar>;
  friend class Hexahedron20<Scalar>;
  friend class Hexahedron26<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;
  using typename Base::Hessian;
  using typename Base::Tensor3;

  static constexpr int kNodes = 27;
  static constexpr int kFaces = 6;

 private:
  std::array<Global, kNodes> global_coords_;
  static const std::array<Local, kNodes> local_coords_;
  static const std::array<std::array<int, 9>, kFaces> faces_;

  using OldShapesOnNewNodes = std::array<std::array<Scalar, 26>, 27 - 26>;
  static OldShapesOnNewNodes old_shapes_on_new_nodes_;
  static OldShapesOnNewNodes BuildOldShapesOnNewNodes() {
    Hexahedron26<Scalar>::old_shapes_on_new_nodes_
        = Hexahedron26<Scalar>::BuildOldShapesOnNewNodes();
    OldShapesOnNewNodes old_shapes_on_new_nodes;
    for (int b = 26; b < 27; ++b) {
      auto &local_b = Hexahedron27<Scalar>::local_coords_[b];
      Hexahedron26<Scalar>::LocalToShapeFunctions(
          local_b[X], local_b[Y], local_b[Z],
          old_shapes_on_new_nodes[b - 26].data());
    }
    return old_shapes_on_new_nodes;
  }

 public:
  int CountNodes() const final {
    return kNodes;
  }
  void SortNodesOnFace(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node) const final {
    assert(9 == face_n_node);
    int i_face = Base::GetFaceId(cell_nodes, face_nodes, face_n_node);
    for (int f = 0; f < face_n_node; ++f) {
      face_nodes[f] = cell_nodes[faces_[i_face][f]];
    }
  }

 public:
  static void LocalToShapeFunctions(Scalar x_local, Scalar y_local,
      Scalar z_local, Scalar *shapes) {
    Scalar factor_x[3] {
      (x_local - 1) * x_local / 2, (1 - x_local * x_local),
      (x_local + 1) * x_local / 2,
    };
    Scalar factor_y[3] {
      (y_local - 1) * y_local / 2, (1 - y_local * y_local),
      (y_local + 1) * y_local / 2,
    };
    Scalar factor_z[3] {
      (z_local - 1) * z_local / 2, (1 - z_local * z_local),
      (z_local + 1) * z_local / 2,
    };
    for (int i = 0; i < kNodes; ++i) {
      auto &local_i = local_coords_[i];
      int i_x = local_i[X] + 1;
      int i_y = local_i[Y] + 1;
      int i_z = local_i[Z] + 1;
      shapes[i] = factor_x[i_x] * factor_y[i_y] * factor_z[i_z];
    }
  }
  std::vector<Scalar> LocalToShapeFunctions(
      Scalar x_local, Scalar y_local, Scalar z_local) const final {
    auto shapes = std::vector<Scalar>(kNodes);
    LocalToShapeFunctions(x_local, y_local, z_local, shapes.data());
    return shapes;
  }
  static void LocalToShapeGradients(Scalar x_local, Scalar y_local,
      Scalar z_local, Local *grads) {
    Scalar factor_x[3] {
      (x_local - 1) * x_local / 2, (1 - x_local * x_local),
      (x_local + 1) * x_local / 2,
    };
    Scalar factor_y[3] {
      (y_local - 1) * y_local / 2, (1 - y_local * y_local),
      (y_local + 1) * y_local / 2,
    };
    Scalar factor_z[3] {
      (z_local - 1) * z_local / 2, (1 - z_local * z_local),
      (z_local + 1) * z_local / 2,
    };
    Scalar grad_x[3] { x_local - 0.5, -2 * x_local, x_local + 0.5 };
    Scalar grad_y[3] { y_local - 0.5, -2 * y_local, y_local + 0.5 };
    Scalar grad_z[3] { z_local - 0.5, -2 * z_local, z_local + 0.5 };
    for (int i = 0; i < kNodes; ++i) {
      auto &local_i = local_coords_[i];
      int i_x = local_i[X] + 1;
      int i_y = local_i[Y] + 1;
      int i_z = local_i[Z] + 1;
      grads[i][X] = grad_x[i_x] * factor_y[i_y] * factor_z[i_z];
      grads[i][Y] = factor_x[i_x] * grad_y[i_y] * factor_z[i_z];
      grads[i][Z] = factor_x[i_x] * factor_y[i_y] * grad_z[i_z];
    }
  }
  std::vector<Local> LocalToShapeGradients(
      Scalar x_local, Scalar y_local, Scalar z_local) const final {
    auto grads = std::vector<Local>(kNodes);
    LocalToShapeGradients(x_local, y_local, z_local, grads.data());
    return grads;
  }

 private:
  static void LocalToNewShapeHessians(Local const &local, Hessian *hessians) {
    Scalar x_local = local[X], y_local = local[Y], z_local = local[Z];
    // local_coords_[i][X] = local_coords_[i][Y] = local_coords_[i][Z] = 0
    Scalar quadratic_x = (1 - x_local * x_local);
    Scalar quadratic_y = (1 - y_local * y_local);
    Scalar quadratic_z = (1 - z_local * z_local);
    // shapes[26] = quadratic_x * quadratic_y * quadratic_z;
    Scalar linear_x = -2 * x_local;
    Scalar linear_y = -2 * y_local;
    Scalar linear_z = -2 * z_local;
    auto &hessian = hessians[26];
    hessian[XX] = -2 * quadratic_y * quadratic_z;
    hessian[YY] = -2 * quadratic_x * quadratic_z;
    hessian[ZZ] = -2 * quadratic_x * quadratic_y;
    hessian[XY] = linear_x * linear_y * quadratic_z;
    hessian[XZ] = linear_x * linear_z * quadratic_y;
    hessian[YZ] = linear_y * linear_z * quadratic_x;
  }
  static void LocalToNewShape3rdOrderDerivatives(Local const &local,
      Tensor3 *tensors) {
    Scalar x_local = local[X], y_local = local[Y], z_local = local[Z];
    // local_coords_[i][X] = local_coords_[i][Y] = local_coords_[i][Z] = 0
    Scalar quadratic_x = (1 - x_local * x_local);
    Scalar quadratic_y = (1 - y_local * y_local);
    Scalar quadratic_z = (1 - z_local * z_local);
    // shapes[26] = quadratic_x * quadratic_y * quadratic_z;
    Scalar linear_x = 4 * x_local;
    Scalar linear_y = 4 * y_local;
    Scalar linear_z = 4 * z_local;
    auto &tensor = tensors[26];
    tensor[XXX] = tensor[YYY] = tensor[ZZZ] = 0;
    tensor[XXY] = linear_y * quadratic_z;
    tensor[XXZ] = linear_z * quadratic_y;
    tensor[YYX] = linear_x * quadratic_z;
    tensor[YYZ] = linear_z * quadratic_x;
    tensor[ZZX] = linear_x * quadratic_y;
    tensor[ZZY] = linear_y * quadratic_x;
    tensor[XYZ] = linear_x * linear_y * linear_z / (-8);
  }

 public:
  static void LocalToShapeHessians(Local const &local,
      Hessian *hessians) {
    Hexahedron26<Scalar>::LocalToShapeHessians(local, hessians);
    LocalToNewShapeHessians(local, hessians);
    for (int b : {26}) {
      auto &old_shapes_on_new_nodes = old_shapes_on_new_nodes_[b - 26];
      for (int a = 0; a < 26; ++a) {
        hessians[a] -= old_shapes_on_new_nodes[a] * hessians[b];
      }
    }
  }
  std::vector<Hessian> LocalToShapeHessians(
        Local const &local) const final {
    auto hessians = std::vector<Hessian>(kNodes);
    LocalToShapeHessians(local, hessians.data());
    return hessians;
  }
  static void LocalToShape3rdOrderDerivatives(Local const &local,
      Tensor3 *tensors) {
    Hexahedron26<Scalar>::LocalToShape3rdOrderDerivatives(local, tensors);
    LocalToNewShape3rdOrderDerivatives(local, tensors);
    for (int b : {26}) {
      auto &old_shapes_on_new_nodes = old_shapes_on_new_nodes_[b - 26];
      for (int a = 0; a < 26; ++a) {
        tensors[a] -= old_shapes_on_new_nodes[a] * tensors[b];
      }
    }
  }
  std::vector<Tensor3> LocalToShape3rdOrderDerivatives(Local const &local)
      const final {
    auto tensors = std::vector<Tensor3>(kNodes);
    LocalToShape3rdOrderDerivatives(local, tensors.data());
    return tensors;
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
  Hexahedron27(std::initializer_list<Global> il) {
    Element<Scalar, 3, 3>::_Build(this, il);
  }
};
template <std::floating_point Scalar>
typename Hexahedron27<Scalar>::OldShapesOnNewNodes
Hexahedron27<Scalar>::old_shapes_on_new_nodes_ =
    Hexahedron27<Scalar>::BuildOldShapesOnNewNodes();

// initialization of static const members:
template <std::floating_point Scalar>
const std::array<typename Hexahedron27<Scalar>::Local, 27>
Hexahedron27<Scalar>::local_coords_{
  // corner nodes on the bottom face
  Hexahedron27::Local(-1, -1, -1), Hexahedron27::Local(+1, -1, -1),
  Hexahedron27::Local(+1, +1, -1), Hexahedron27::Local(-1, +1, -1),
  // corner nodes on the top face
  Hexahedron27::Local(-1, -1, +1), Hexahedron27::Local(+1, -1, +1),
  Hexahedron27::Local(+1, +1, +1), Hexahedron27::Local(-1, +1, +1),
  // mid-edge nodes on the bottom face
  Hexahedron27::Local(0, -1, -1), Hexahedron27::Local(+1, 0, -1),
  Hexahedron27::Local(0, +1, -1), Hexahedron27::Local(-1, 0, -1),
  // mid-edge nodes on vertical edges
  Hexahedron27::Local(-1, -1, 0), Hexahedron27::Local(+1, -1, 0),
  Hexahedron27::Local(+1, +1, 0), Hexahedron27::Local(-1, +1, 0),
  // mid-edge nodes on the top face
  Hexahedron27::Local(0, -1, +1), Hexahedron27::Local(+1, 0, +1),
  Hexahedron27::Local(0, +1, +1), Hexahedron27::Local(-1, 0, +1),
  // mid-face nodes
  Hexahedron27::Local(0, 0, -1),
  Hexahedron27::Local(0, -1, 0), Hexahedron27::Local(+1, 0, 0),
  Hexahedron27::Local(0, +1, 0), Hexahedron27::Local(-1, 0, 0),
  Hexahedron27::Local(0, 0, +1),
  // center node
  Hexahedron27::Local(0, 0, 0),
};
template <std::floating_point Scalar>
const std::array<std::array<int, 9>, 6>
Hexahedron27<Scalar>::faces_{
  /* See http://cgns.github.io/CGNS_docs_current/sids/conv.figs/hexa_27.png for node numbering.
   * Faces can be distinguished by the sum of the three minimum node ids.
   */
  /* zeta = -1 */0, 3, 2, 1, 11, 10, 9, 8, 20,
  /*  eta = -1 */0, 1, 5, 4, 8, 13, 16, 12, 21,
  /*   xi = +1 */1, 2, 6, 5, 9, 14, 17, 13, 22,
  /*  eta = +1 */2, 3, 7, 6, 10, 15, 18, 14, 23,
  /*   xi = -1 */0, 4, 7, 3, 12, 19, 15, 11, 24,
  /* zeta = +1 */4, 5, 6, 7, 16, 17, 18, 19, 25,
};

/**
 * @brief Coordinate map on 8-node hexahedral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Hexahedron8 : public Hexahedron<Scalar> {
  using Base = Hexahedron<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;
  using typename Base::Hessian;
  using typename Base::Tensor3;

  static constexpr int kNodes = 8;

 private:
  std::array<Global, kNodes> global_coords_;

 public:
  int CountNodes() const final {
    return kNodes;
  }
  void SortNodesOnFace(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node) const final {
    assert(4 == face_n_node);
    int i_face = Base::GetFaceId(cell_nodes, face_nodes, face_n_node);
    for (int f = 0; f < face_n_node; ++f) {
      face_nodes[f] = cell_nodes[Hexahedron27<Scalar>::faces_[i_face][f]];
    }
  }

 public:
  static void LocalToShapeFunctions(Scalar x_local, Scalar y_local,
      Scalar z_local, Scalar *shapes) {
    for (int i = 0; i < kNodes; ++i) {
      auto &local_i = Hexahedron27<Scalar>::local_coords_[i];
      auto val = (1 + local_i[X] * x_local);
          val *= (1 + local_i[Y] * y_local);
          val *= (1 + local_i[Z] * z_local);
      shapes[i] = val / 8;
    }
  }
  std::vector<Scalar> LocalToShapeFunctions(
      Scalar x_local, Scalar y_local, Scalar z_local) const final {
    auto shapes = std::vector<Scalar>(kNodes);
    LocalToShapeFunctions(x_local, y_local, z_local, shapes.data());
    return shapes;
  }
  static void LocalToShapeGradients(Scalar x_local, Scalar y_local,
      Scalar z_local, Local *grads) {
    std::array<Scalar, kNodes> factor_xy, factor_yz, factor_zx;
    for (int i = 0; i < kNodes; ++i) {
      auto &local_i = Hexahedron27<Scalar>::local_coords_[i];
      auto factor_x = (1 + local_i[X] * x_local);
      auto factor_y = (1 + local_i[Y] * y_local);
      auto factor_z = (1 + local_i[Z] * z_local);
      factor_xy[i] = factor_x * factor_y / 8;
      factor_yz[i] = factor_y * factor_z / 8;
      factor_zx[i] = factor_z * factor_x / 8;
    }
    for (int i = 0; i < kNodes; ++i) {
      auto &local_i = Hexahedron27<Scalar>::local_coords_[i];
      auto &grad_i = grads[i];
      grad_i[X] = local_i[X] * factor_yz[i];
      grad_i[Y] = local_i[Y] * factor_zx[i];
      grad_i[Z] = local_i[Z] * factor_xy[i];
    }
  }
  std::vector<Local> LocalToShapeGradients(
      Scalar x_local, Scalar y_local, Scalar z_local) const final {
    auto grads = std::vector<Local>(kNodes);
    LocalToShapeGradients(x_local, y_local, z_local, grads.data());
    return grads;
  }
  static void LocalToShapeHessians(Local const &local, Hessian *hessians) {
    Scalar x_local = local[X], y_local = local[Y], z_local = local[Z];
    std::array<Scalar, kNodes> factor_x, factor_y, factor_z;
    for (int i = 0; i < kNodes; ++i) {
      auto &local_i = Hexahedron27<Scalar>::local_coords_[i];
      factor_x[i] = (1 + local_i[X] * x_local) / 8;
      factor_y[i] = (1 + local_i[Y] * y_local) / 8;
      factor_z[i] = (1 + local_i[Z] * z_local) / 8;
    }
    for (int i = 0; i < kNodes; ++i) {
      auto &local_i = Hexahedron27<Scalar>::local_coords_[i];
      auto &hessian_i = hessians[i];
      hessian_i[XX] = hessian_i[YY] = hessian_i[ZZ] = 0;
      hessian_i[XY] = local_i[X] * local_i[Y] * factor_z[i];
      hessian_i[XZ] = local_i[X] * local_i[Z] * factor_y[i];
      hessian_i[YZ] = local_i[Y] * local_i[Z] * factor_x[i];
    }
  }
  std::vector<Hessian> LocalToShapeHessians(Local const &local) const final {
    auto hessians = std::vector<Hessian>(kNodes);
    LocalToShapeHessians(local, hessians.data());
    return hessians;
  }
  static void LocalToShape3rdOrderDerivatives(Local const &local,
      Tensor3 *tensors) {
    for (int i = 0; i < kNodes; ++i) {
      auto &local_i = Hexahedron27<Scalar>::local_coords_[i];
      auto &tensor_i = tensors[i];
      tensor_i.setZero();
      tensor_i[XYZ] = local_i[X] * local_i[Y] * local_i[Z] / 8;
    }
  }
  std::vector<Tensor3> LocalToShape3rdOrderDerivatives(Local const &local)
      const final {
    auto tensors = std::vector<Tensor3>(kNodes);
    LocalToShape3rdOrderDerivatives(local, tensors.data());
    return tensors;
  }

 public:
  Global const &GetGlobalCoord(int i) const final {
    assert(0 <= i && i < CountNodes());
    return global_coords_[i];
  }
  Local const &GetLocalCoord(int i) const final {
    assert(0 <= i && i < CountNodes());
    return Hexahedron27<Scalar>::local_coords_[i];
  }

 public:
  Hexahedron8(std::initializer_list<Global> il) {
    Element<Scalar, 3, 3>::_Build(this, il);
  }
};

/**
 * @brief Coordinate map on 20-node hexahedral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Hexahedron20 : public Hexahedron<Scalar> {
  using Base = Hexahedron<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;
  using typename Base::Hessian;
  using typename Base::Tensor3;

  static constexpr int kNodes = 20;

 private:
  std::array<Global, kNodes> global_coords_;

  using OldShapesOnNewNodes = std::array<std::array<Scalar, 8>, 20 - 8>;
  static OldShapesOnNewNodes old_shapes_on_new_nodes_;
  static OldShapesOnNewNodes BuildOldShapesOnNewNodes() {
    OldShapesOnNewNodes old_shapes_on_new_nodes;
    for (int b = 8; b < 20; ++b) {
      auto &local_b = Hexahedron27<Scalar>::local_coords_[b];
      Hexahedron8<Scalar>::LocalToShapeFunctions(
          local_b[X], local_b[Y], local_b[Z],
          old_shapes_on_new_nodes[b - 8].data());
    }
    return old_shapes_on_new_nodes;
  }
  friend class Hexahedron26<Scalar>;

 public:
  int CountNodes() const final {
    return kNodes;
  }
  void SortNodesOnFace(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node) const final {
    assert(8 == face_n_node);
    int i_face = Base::GetFaceId(cell_nodes, face_nodes, face_n_node);
    for (int f = 0; f < face_n_node; ++f) {
      face_nodes[f] = cell_nodes[Hexahedron27<Scalar>::faces_[i_face][f]];
    }
  }

 private:
  static void LocalToNewShapeFunctions(Scalar x_local, Scalar y_local,
      Scalar z_local, Scalar *shapes) {
    // local_coords_[i][X] = 0
    Scalar factor_x = (1 - x_local * x_local) / 4;
    for (int a : {8, 10, 16, 18}) {
      shapes[a] = factor_x
          * (1 + Hexahedron27<Scalar>::local_coords_[a][Y] * y_local)
          * (1 + Hexahedron27<Scalar>::local_coords_[a][Z] * z_local);
    }
    // local_coords_[i][Y] = 0
    Scalar factor_y = (1 - y_local * y_local) / 4;
    for (int a : {9, 11, 17, 19}) {
      shapes[a] = factor_y
          * (1 + Hexahedron27<Scalar>::local_coords_[a][X] * x_local)
          * (1 + Hexahedron27<Scalar>::local_coords_[a][Z] * z_local);
    }
    // local_coords_[i][Z] = 0
    Scalar factor_z = (1 - z_local * z_local) / 4;
    for (int a : {12, 13, 14, 15}) {
      shapes[a] = factor_z
          * (1 + Hexahedron27<Scalar>::local_coords_[a][X] * x_local)
          * (1 + Hexahedron27<Scalar>::local_coords_[a][Y] * y_local);
    }
  }
  static void LocalToNewShapeGradients(Scalar x_local, Scalar y_local,
      Scalar z_local, Local *grads) {
    // local_coords_[i][X] = 0
    Scalar shape_x = (1 - x_local * x_local) / 4;
    Scalar grad_x = -x_local / 2;
    for (int a : {8, 10, 16, 18}) {
      Scalar y_a = Hexahedron27<Scalar>::local_coords_[a][Y];
      Scalar z_a = Hexahedron27<Scalar>::local_coords_[a][Z];
      Scalar shape_y = (1 + y_a * y_local);
      Scalar shape_z = (1 + z_a * z_local);
      grads[a][X] = grad_x * shape_y * shape_z;
      grads[a][Y] = shape_x * y_a * shape_z;
      grads[a][Z] = shape_x * shape_y * z_a;
    }
    // local_coords_[i][Y] = 0
    Scalar shape_y = (1 - y_local * y_local) / 4;
    Scalar grad_y = -y_local / 2;
    for (int a : {9, 11, 17, 19}) {
      Scalar x_a = Hexahedron27<Scalar>::local_coords_[a][X];
      Scalar z_a = Hexahedron27<Scalar>::local_coords_[a][Z];
      Scalar shape_x = (1 + x_a * x_local);
      Scalar shape_z = (1 + z_a * z_local);
      grads[a][X] = x_a * shape_y * shape_z;
      grads[a][Y] = shape_x * grad_y * shape_z;
      grads[a][Z] = shape_x * shape_y * z_a;
    }
    // local_coords_[i][Z] = 0
    Scalar shape_z = (1 - z_local * z_local) / 4;
    Scalar grad_z = -z_local / 2;
    for (int a : {12, 13, 14, 15}) {
      Scalar x_a = Hexahedron27<Scalar>::local_coords_[a][X];
      Scalar y_a = Hexahedron27<Scalar>::local_coords_[a][Y];
      Scalar shape_x = (1 + x_a * x_local);
      Scalar shape_y = (1 + y_a * y_local);
      grads[a][X] = x_a * shape_y * shape_z;
      grads[a][Y] = shape_x * y_a * shape_z;
      grads[a][Z] = shape_x * shape_y * grad_z;
    }
  }
  static void LocalToNewShapeHessians(Local const &local, Hessian *hessians) {
    Scalar x_local = local[X], y_local = local[Y], z_local = local[Z];
    // local_coords_[i][X] = 0
    for (int a : {8, 10, 16, 18}) {
      auto &local_a = Hexahedron27<Scalar>::local_coords_[a];
      auto &hessian = hessians[a];
      auto factor_x = -0.5 * x_local;
      auto factor_y = (1 + local_a[Y] * y_local);
      auto factor_z = (1 + local_a[Z] * z_local);
      hessian[XX] = -0.5 * factor_y * factor_z;
      hessian[XY] = factor_x * local_a[Y] * factor_z;
      hessian[XZ] = factor_x * factor_y * local_a[Z];
      hessian[YZ] = (0.25 - factor_x * factor_x) * local_a[Y] * local_a[Z];
      hessian[YY] = hessian[ZZ] = 0;
    }
    // local_coords_[i][Y] = 0
    for (int a : {9, 11, 17, 19}) {
      auto &local_a = Hexahedron27<Scalar>::local_coords_[a];
      auto &hessian = hessians[a];
      auto factor_x = (1 + local_a[X] * x_local);
      auto factor_y = -0.5 * y_local;
      auto factor_z = (1 + local_a[Z] * z_local);
      hessian[YY] = -0.5 * factor_x * factor_z;
      hessian[YX] = factor_y * local_a[X] * factor_z;
      hessian[YZ] = factor_y * factor_x * local_a[Z];
      hessian[XZ] = (0.25 - factor_y * factor_y) * local_a[X] * local_a[Z];
      hessian[XX] = hessian[ZZ] = 0;
    }
    // local_coords_[i][Z] = 0
    for (int a : {12, 13, 14, 15}) {
      auto &local_a = Hexahedron27<Scalar>::local_coords_[a];
      auto &hessian = hessians[a];
      auto factor_x = (1 + local_a[X] * x_local);
      auto factor_y = (1 + local_a[Y] * y_local);
      auto factor_z = -0.5 * z_local;
      hessian[ZZ] = -0.5 * factor_x * factor_y;
      hessian[ZX] = factor_z * local_a[X] * factor_y;
      hessian[ZY] = factor_z * factor_x * local_a[Y];
      hessian[XY] = (0.25 - factor_z * factor_z) * local_a[X] * local_a[Y];
      hessian[XX] = hessian[YY] = 0;
    }
  }
  static void LocalToNewShape3rdOrderDerivatives(Local const &local,
      Tensor3 *tensors) {
    Scalar x_local = local[X], y_local = local[Y], z_local = local[Z];
    // local_coords_[i][X] = 0
    for (int a : {8, 10, 16, 18}) {
      auto &local_a = Hexahedron27<Scalar>::local_coords_[a];
      auto &tensor = tensors[a];
      tensor.setZero();
      tensor[XXY] = -0.5 * local_a[Y] * (1 + local_a[Z] * z_local);
      tensor[XXZ] = -0.5 * (1 + local_a[Y] * y_local) * local_a[Z];
      tensor[XYZ] = -0.5 * x_local * local_a[Y] * local_a[Z];
    }
    // local_coords_[i][Y] = 0
    for (int a : {9, 11, 17, 19}) {
      auto &local_a = Hexahedron27<Scalar>::local_coords_[a];
      auto &tensor = tensors[a];
      tensor.setZero();
      tensor[YYX] = -0.5 * local_a[X] * (1 + local_a[Z] * z_local);
      tensor[YYZ] = -0.5 * (1 + local_a[X] * x_local) * local_a[Z];
      tensor[XYZ] = -0.5 * y_local * local_a[X] * local_a[Z];
    }
    // local_coords_[i][Z] = 0
    for (int a : {12, 13, 14, 15}) {
      auto &local_a = Hexahedron27<Scalar>::local_coords_[a];
      auto &tensor = tensors[a];
      tensor.setZero();
      tensor[ZZX] = -0.5 * local_a[X] * (1 + local_a[Y] * y_local);
      tensor[ZZY] = -0.5 * (1 + local_a[X] * x_local) * local_a[Y];
      tensor[XYZ] = -0.5 * z_local * local_a[X] * local_a[Y];
    }
  }

 public:
  static void LocalToShapeFunctions(Scalar x_local, Scalar y_local,
      Scalar z_local, Scalar *shapes) {
    Hexahedron8<Scalar>::LocalToShapeFunctions(
        x_local, y_local, z_local, shapes);
    LocalToNewShapeFunctions(x_local, y_local, z_local, shapes);
    for (int b = 8; b < 20; ++b) {
      auto &old_shapes_on_new_nodes = old_shapes_on_new_nodes_[b - 8];
      for (int a = 0; a < 8; ++a) {
        shapes[a] -= old_shapes_on_new_nodes[a] * shapes[b];
      }
    }
  }
  std::vector<Scalar> LocalToShapeFunctions(
      Scalar x_local, Scalar y_local, Scalar z_local) const final {
    auto shapes = std::vector<Scalar>(kNodes);
    LocalToShapeFunctions(x_local, y_local, z_local, shapes.data());
    return shapes;
  }
  static void LocalToShapeGradients(Scalar x_local, Scalar y_local,
      Scalar z_local, Local *grads) {
    Hexahedron8<Scalar>::LocalToShapeGradients(
        x_local, y_local, z_local, grads);
    LocalToNewShapeGradients(x_local, y_local, z_local, grads);
    for (int b = 8; b < 20; ++b) {
      auto &old_shapes_on_new_nodes = old_shapes_on_new_nodes_[b - 8];
      for (int a = 0; a < 8; ++a) {
        grads[a] -= old_shapes_on_new_nodes[a] * grads[b];
      }
    }
  }
  std::vector<Local> LocalToShapeGradients(
      Scalar x_local, Scalar y_local, Scalar z_local) const final {
    auto grads = std::vector<Local>(kNodes);
    LocalToShapeGradients(x_local, y_local, z_local, grads.data());
    return grads;
  }
  static void LocalToShapeHessians(Local const &local,
      Hessian *hessians) {
    Hexahedron8<Scalar>::LocalToShapeHessians(local, hessians);
    LocalToNewShapeHessians(local, hessians);
    for (int b = 8; b < 20; ++b) {
      auto &old_shapes_on_new_nodes = old_shapes_on_new_nodes_[b - 8];
      for (int a = 0; a < 8; ++a) {
        hessians[a] -= old_shapes_on_new_nodes[a] * hessians[b];
      }
    }
  }
  std::vector<Hessian> LocalToShapeHessians(
        Local const &local) const final {
    auto hessians = std::vector<Hessian>(kNodes);
    LocalToShapeHessians(local, hessians.data());
    return hessians;
  }
  static void LocalToShape3rdOrderDerivatives(Local const &local,
      Tensor3 *tensors) {
    Hexahedron8<Scalar>::LocalToShape3rdOrderDerivatives(local, tensors);
    LocalToNewShape3rdOrderDerivatives(local, tensors);
    for (int b = 8; b < 20; ++b) {
      auto &old_shapes_on_new_nodes = old_shapes_on_new_nodes_[b - 8];
      for (int a = 0; a < 8; ++a) {
        tensors[a] -= old_shapes_on_new_nodes[a] * tensors[b];
      }
    }
  }
  std::vector<Tensor3> LocalToShape3rdOrderDerivatives(Local const &local)
      const final {
    auto tensors = std::vector<Tensor3>(kNodes);
    LocalToShape3rdOrderDerivatives(local, tensors.data());
    return tensors;
  }

 public:
  Global const &GetGlobalCoord(int i) const final {
    assert(0 <= i && i < CountNodes());
    return global_coords_[i];
  }
  Local const &GetLocalCoord(int i) const final {
    assert(0 <= i && i < CountNodes());
    return Hexahedron27<Scalar>::local_coords_[i];
  }

 public:
  Hexahedron20(std::initializer_list<Global> il) {
    Element<Scalar, 3, 3>::_Build(this, il);
  }
};
template <std::floating_point Scalar>
typename Hexahedron20<Scalar>::OldShapesOnNewNodes
Hexahedron20<Scalar>::old_shapes_on_new_nodes_ =
    Hexahedron20<Scalar>::BuildOldShapesOnNewNodes();

/**
 * @brief Coordinate map on 26-node hexahedral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Hexahedron26 : public Hexahedron<Scalar> {
  using Base = Hexahedron<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;
  using typename Base::Hessian;
  using typename Base::Tensor3;

  static constexpr int kNodes = 26;

 private:
  std::array<Global, kNodes> global_coords_;

  using OldShapesOnNewNodes = std::array<std::array<Scalar, 20>, 26 - 20>;
  static OldShapesOnNewNodes old_shapes_on_new_nodes_;
  static OldShapesOnNewNodes BuildOldShapesOnNewNodes() {
    Hexahedron20<Scalar>::old_shapes_on_new_nodes_
        = Hexahedron20<Scalar>::BuildOldShapesOnNewNodes();
    OldShapesOnNewNodes old_shapes_on_new_nodes;
    for (int b = 20; b < 26; ++b) {
      auto &local_b = Hexahedron27<Scalar>::local_coords_[b];
      Hexahedron20<Scalar>::LocalToShapeFunctions(
          local_b[X], local_b[Y], local_b[Z],
          old_shapes_on_new_nodes[b - 20].data());
    }
    return old_shapes_on_new_nodes;
  }
  friend class Hexahedron27<Scalar>;

 public:
  int CountNodes() const final {
    return kNodes;
  }
  void SortNodesOnFace(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node) const final {
    assert(9 == face_n_node);
    int i_face = Base::GetFaceId(cell_nodes, face_nodes, face_n_node);
    for (int f = 0; f < face_n_node; ++f) {
      face_nodes[f] = cell_nodes[Hexahedron27<Scalar>::faces_[i_face][f]];
    }
  }

 private:
  static void LocalToNewShapeFunctions(Scalar x_local, Scalar y_local,
      Scalar z_local, Scalar *shapes) {
    Scalar quadratic_x = (1 - x_local * x_local);
    Scalar quadratic_y = (1 - y_local * y_local);
    Scalar quadratic_z = (1 - z_local * z_local);
    // local_coords_[i][Y] = local_coords_[i][Z] = 0
    Scalar quadratic_yz = quadratic_y * quadratic_z / 2;
    for (int a : {22, 24}) {
      shapes[a] = quadratic_yz
          * (1 + Hexahedron27<Scalar>::local_coords_[a][X] * x_local);
    }
    // local_coords_[i][Z] = local_coords_[i][X] = 0
    Scalar quadratic_zx = quadratic_z * quadratic_x / 2;
    for (int a : {21, 23}) {
      shapes[a] = quadratic_zx
          * (1 + Hexahedron27<Scalar>::local_coords_[a][Y] * y_local);
    }
    // local_coords_[i][X] = local_coords_[i][Y] = 0
    Scalar quadratic_xy = quadratic_x * quadratic_y / 2;
    for (int a : {20, 25}) {
      shapes[a] = quadratic_xy
          * (1 + Hexahedron27<Scalar>::local_coords_[a][Z] * z_local);
    }
  }
  static void LocalToNewShapeGradients(Scalar x_local, Scalar y_local,
      Scalar z_local, Local *grads) {
    Scalar quadratic_x = (1 - x_local * x_local);
    Scalar quadratic_y = (1 - y_local * y_local);
    Scalar quadratic_z = (1 - z_local * z_local);
    // local_coords_[i][Y] = local_coords_[i][Z] = 0
    for (int a : {22, 24}) {
      Scalar factor_x = (1
          + Hexahedron27<Scalar>::local_coords_[a][X] * x_local);
      grads[a][X] = (Hexahedron27<Scalar>::local_coords_[a][X] / 2)
          * quadratic_y * quadratic_z;
      grads[a][Y] = factor_x * (-y_local) * quadratic_z;
      grads[a][Z] = factor_x * (-z_local) * quadratic_y;
    }
    // local_coords_[i][Z] = local_coords_[i][X] = 0
    for (int a : {21, 23}) {
      Scalar factor_y = (1
          + Hexahedron27<Scalar>::local_coords_[a][Y] * y_local);
      grads[a][Y] = (Hexahedron27<Scalar>::local_coords_[a][Y] / 2)
          * quadratic_z * quadratic_x;
      grads[a][Z] = factor_y * (-z_local) * quadratic_x;
      grads[a][X] = factor_y * (-x_local) * quadratic_z;
    }
    // local_coords_[i][X] = local_coords_[i][Y] = 0
    for (int a : {20, 25}) {
      Scalar factor_z = (1
          + Hexahedron27<Scalar>::local_coords_[a][Z] * z_local);
      grads[a][Z] = (Hexahedron27<Scalar>::local_coords_[a][Z] / 2)
          * quadratic_x * quadratic_y;
      grads[a][X] = factor_z * (-x_local) * quadratic_y;
      grads[a][Y] = factor_z * (-y_local) * quadratic_x;
    }
  }
  static void LocalToNewShapeHessians(Local const &local, Hessian *hessians) {
    Scalar x_local = local[X], y_local = local[Y], z_local = local[Z];
    Scalar quadratic_x = (1 - x_local * x_local);
    Scalar quadratic_y = (1 - y_local * y_local);
    Scalar quadratic_z = (1 - z_local * z_local);
    // local_coords_[i][Y] = local_coords_[i][Z] = 0
    for (int a : {22, 24}) {
      auto &local_a = Hexahedron27<Scalar>::local_coords_[a];
      auto &hessian = hessians[a];
      auto linear_x = (1 + local_a[X] * x_local);
      // shapes[a] = linear_x * quadratic_y * quadratic_z;
      hessian[XX] = 0;
      hessian[XY] = local_a[X] * (-y_local) * quadratic_z;
      hessian[XZ] = local_a[X] * (-z_local) * quadratic_y;
      hessian[YY] = linear_x * (-quadratic_z);
      hessian[YZ] = linear_x * y_local * z_local * 2;
      hessian[ZZ] = linear_x * (-quadratic_y);
    }
    // local_coords_[i][Z] = local_coords_[i][X] = 0
    for (int a : {21, 23}) {
      auto &local_a = Hexahedron27<Scalar>::local_coords_[a];
      auto &hessian = hessians[a];
      auto linear_y = (1 + local_a[Y] * y_local);
      // shapes[a] = quadratic_x * linear_y * quadratic_z;
      hessian[YY] = 0;
      hessian[YX] = local_a[Y] * (-x_local) * quadratic_z;
      hessian[YZ] = local_a[Y] * (-z_local) * quadratic_x;
      hessian[XX] = linear_y * (-quadratic_z);
      hessian[XZ] = linear_y * x_local * z_local * 2;
      hessian[ZZ] = linear_y * (-quadratic_x);
    }
    // local_coords_[i][X] = local_coords_[i][Y] = 0
    for (int a : {20, 25}) {
      auto &local_a = Hexahedron27<Scalar>::local_coords_[a];
      auto &hessian = hessians[a];
      auto linear_z = (1 + local_a[Z] * z_local);
      // shapes[a] = quadratic_x * quadratic_y * linear_z;
      hessian[ZZ] = 0;
      hessian[ZX] = local_a[Z] * (-x_local) * quadratic_y;
      hessian[ZY] = local_a[Z] * (-y_local) * quadratic_x;
      hessian[XX] = linear_z * (-quadratic_y);
      hessian[XY] = linear_z * x_local * y_local * 2;
      hessian[YY] = linear_z * (-quadratic_x);
    }
  }
  static void LocalToNewShape3rdOrderDerivatives(Local const &local,
      Tensor3 *tensors) {
    Scalar x_local = local[X], y_local = local[Y], z_local = local[Z];
    Scalar quadratic_x = (1 - x_local * x_local);
    Scalar quadratic_y = (1 - y_local * y_local);
    Scalar quadratic_z = (1 - z_local * z_local);
    // local_coords_[i][Y] = local_coords_[i][Z] = 0
    for (int a : {22, 24}) {
      // shapes[a] = linear_x * quadratic_y * quadratic_z;
      auto &local_a = Hexahedron27<Scalar>::local_coords_[a];
      auto &tensor = tensors[a];
      tensor[XXX] = tensor[YYY] = tensor[ZZZ]
                  = tensor[XXY] = tensor[XXZ] = 0;
      tensor[XYY] = -local_a[X] * quadratic_z;
      tensor[XZZ] = -local_a[X] * quadratic_y;
      tensor[XYZ] = local_a[X] * y_local * z_local * 2;
      auto linear_x = 1 + local_a[X] * x_local;
      tensor[YYZ] = linear_x * z_local * 2;
      tensor[ZZY] = linear_x * y_local * 2;
    }
    // local_coords_[i][Z] = local_coords_[i][X] = 0
    for (int a : {21, 23}) {
      // shapes[a] = quadratic_x * linear_y * quadratic_z;
      auto &local_a = Hexahedron27<Scalar>::local_coords_[a];
      auto &tensor = tensors[a];
      tensor[XXX] = tensor[YYY] = tensor[ZZZ]
                  = tensor[YYX] = tensor[YYZ] = 0;
      tensor[YXX] = -local_a[Y] * quadratic_z;
      tensor[YZZ] = -local_a[Y] * quadratic_x;
      tensor[XYZ] = local_a[Y] * x_local * z_local * 2;
      auto linear_y = 1 + local_a[Y] * y_local;
      tensor[XXZ] = linear_y * z_local * 2;
      tensor[ZZX] = linear_y * x_local * 2;
    }
    // local_coords_[i][X] = local_coords_[i][Y] = 0
    for (int a : {20, 25}) {
      // shapes[a] = quadratic_x * quadratic_y * linear_z;
      auto &local_a = Hexahedron27<Scalar>::local_coords_[a];
      auto &tensor = tensors[a];
      tensor[XXX] = tensor[YYY] = tensor[ZZZ]
                  = tensor[ZZX] = tensor[ZZY] = 0;
      tensor[ZXX] = -local_a[Z] * quadratic_y;
      tensor[ZYY] = -local_a[Z] * quadratic_x;
      tensor[XYZ] = local_a[Z] * x_local * y_local * 2;
      auto linear_z = 1 + local_a[Z] * z_local;
      tensor[XXY] = linear_z * y_local * 2;
      tensor[YYX] = linear_z * x_local * 2;
    }
  }

 public:
  static void LocalToShapeFunctions(Scalar x_local, Scalar y_local,
      Scalar z_local, Scalar *shapes) {
    Hexahedron20<Scalar>::LocalToShapeFunctions(
        x_local, y_local, z_local, shapes);
    LocalToNewShapeFunctions(x_local, y_local, z_local, shapes);
    for (int b = 20; b < 26; ++b) {
      auto &old_shapes_on_new_nodes = old_shapes_on_new_nodes_[b - 20];
      for (int a = 0; a < 20; ++a) {
        shapes[a] -= old_shapes_on_new_nodes[a] * shapes[b];
      }
    }
  }
  std::vector<Scalar> LocalToShapeFunctions(
      Scalar x_local, Scalar y_local, Scalar z_local) const final {
    auto shapes = std::vector<Scalar>(kNodes);
    LocalToShapeFunctions(x_local, y_local, z_local, shapes.data());
    return shapes;
  }
  static void LocalToShapeGradients(Scalar x_local, Scalar y_local,
      Scalar z_local, Local *grads) {
    Hexahedron20<Scalar>::LocalToShapeGradients(
        x_local, y_local, z_local, grads);
    LocalToNewShapeGradients(x_local, y_local, z_local, grads);
    for (int b = 20; b < 26; ++b) {
      auto &old_shapes_on_new_nodes = old_shapes_on_new_nodes_[b - 20];
      for (int a = 0; a < 20; ++a) {
        grads[a] -= old_shapes_on_new_nodes[a] * grads[b];
      }
    }
  }
  std::vector<Local> LocalToShapeGradients(
      Scalar x_local, Scalar y_local, Scalar z_local) const final {
    auto grads = std::vector<Local>(kNodes);
    LocalToShapeGradients(x_local, y_local, z_local, grads.data());
    return grads;
  }
  static void LocalToShapeHessians(Local const &local,
      Hessian *hessians) {
    Hexahedron20<Scalar>::LocalToShapeHessians(local, hessians);
    LocalToNewShapeHessians(local, hessians);
    for (int b = 20; b < 26; ++b) {
      auto &old_shapes_on_new_nodes = old_shapes_on_new_nodes_[b - 20];
      for (int a = 0; a < 20; ++a) {
        hessians[a] -= old_shapes_on_new_nodes[a] * hessians[b];
      }
    }
  }
  std::vector<Hessian> LocalToShapeHessians(
        Local const &local) const final {
    auto hessians = std::vector<Hessian>(kNodes);
    LocalToShapeHessians(local, hessians.data());
    return hessians;
  }
  static void LocalToShape3rdOrderDerivatives(Local const &local,
      Tensor3 *tensors) {
    Hexahedron20<Scalar>::LocalToShape3rdOrderDerivatives(local, tensors);
    LocalToNewShape3rdOrderDerivatives(local, tensors);
    for (int b = 20; b < 26; ++b) {
      auto &old_shapes_on_new_nodes = old_shapes_on_new_nodes_[b - 20];
      for (int a = 0; a < 20; ++a) {
        tensors[a] -= old_shapes_on_new_nodes[a] * tensors[b];
      }
    }
  }
  std::vector<Tensor3> LocalToShape3rdOrderDerivatives(Local const &local)
      const final {
    auto tensors = std::vector<Tensor3>(kNodes);
    LocalToShape3rdOrderDerivatives(local, tensors.data());
    return tensors;
  }

 public:
  Global const &GetGlobalCoord(int i) const final {
    assert(0 <= i && i < CountNodes());
    return global_coords_[i];
  }
  Local const &GetLocalCoord(int i) const final {
    assert(0 <= i && i < CountNodes());
    return Hexahedron27<Scalar>::local_coords_[i];
  }

 public:
  Hexahedron26(std::initializer_list<Global> il) {
    Element<Scalar, 3, 3>::_Build(this, il);
  }
};
template <std::floating_point Scalar>
typename Hexahedron26<Scalar>::OldShapesOnNewNodes
Hexahedron26<Scalar>::old_shapes_on_new_nodes_ =
    Hexahedron26<Scalar>::BuildOldShapesOnNewNodes();

}  // namespace geometry
}  // namespace mini

#endif  // MINI_GEOMETRY_HEXAHEDRON_HPP_
