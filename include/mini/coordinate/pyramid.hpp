//  Copyright 2023 PEI Weicheng
#ifndef MINI_GEOMETRY_PYRAMID_HPP_
#define MINI_GEOMETRY_PYRAMID_HPP_

#include <concepts>

#include <cassert>

#include <array>
#include <initializer_list>
#include <numeric>
#include <vector>

#include "mini/geometry/element.hpp"
#include "mini/geometry/cell.hpp"

namespace mini {
namespace geometry {

/**
 * @brief Abstract coordinate map on pyramidal elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Pyramid : public Cell<Scalar> {
  using Base = Cell<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  int CountCorners() const final {
    return 5;
  }
  const Global &center() const final {
    return center_;
  }

 protected:
  Global center_;
  void _BuildCenter() final {
    center_ = this->LocalToGlobal(0, 0, -0.5);
  }

  static int GetFaceId(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node/* number of nodes on triangle */) {
    int cnt[5] = { 0, 0, 0, 0, 1 };
    for (int f = 0; f < face_n_node; ++f) {
      for (int c = 0; c < 4; ++c) {
        if (cell_nodes[c] == face_nodes[f]) {
          cnt[c] = 1;
          break;
        }
      }
    }
    assert(3 == std::accumulate(cnt, cnt + 5, 0));
    int i_face;
    if (cnt[0]) {
      if (cnt[1]) {
        i_face = 0;
      } else {
        assert(cnt[3]);
        i_face = 3;
      }
    } else {
      assert(cnt[2]);
      if (cnt[1]) {
        i_face = 1;
      } else {
        assert(cnt[3]);
        i_face = 2;
      }
    }
    return i_face;
  }
};

/**
 * @brief Coordinate map on 5-node pyramidal elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Pyramid5 : public Pyramid<Scalar> {
  using Base = Pyramid<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  static constexpr int kNodes = 5;

 private:
  std::array<Global, kNodes> global_coords_;
  static const std::array<Local, kNodes> local_coords_;
  static const std::array<std::array<int, 3>, 4> faces_;

 public:
  int CountNodes() const final {
    return kNodes;
  }
  void SortNodesOnFace(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node) const final {
    if (face_n_node == 3) {
      int i_face = Base::GetFaceId(cell_nodes, face_nodes, face_n_node);
      for (int f = 0; f < face_n_node; ++f) {
        face_nodes[f] = cell_nodes[faces_[i_face][f]];
      }
    } else {
      // the face is the bottom
      face_nodes[0] = cell_nodes[0];
      face_nodes[1] = cell_nodes[3];
      face_nodes[2] = cell_nodes[2];
      face_nodes[3] = cell_nodes[1];
    }
  }

 public:
  static void LocalToShapeFunctions(Scalar x_local, Scalar y_local,
      Scalar z_local, Scalar *shapes) {
    for (int i = 0; i < 4; ++i) {
      auto &local_i = local_coords_[i];
      auto val = (1 + local_i[X] * x_local);
          val *= (1 + local_i[Y] * y_local);
          val *= (1 + local_i[Z] * z_local);
      shapes[i] = val / 8;
    }
    shapes[4] = (1 + z_local) / 2;
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
    for (int i = 0; i < 4; ++i) {
      auto &local_i = local_coords_[i];
      auto factor_x = (1 + local_i[X] * x_local);
      auto factor_y = (1 + local_i[Y] * y_local);
      auto factor_z = (1 + local_i[Z] * z_local);
      factor_xy[i] = factor_x * factor_y / 8;
      factor_yz[i] = factor_y * factor_z / 8;
      factor_zx[i] = factor_z * factor_x / 8;
    }
    for (int i = 0; i < 4; ++i) {
      auto &local_i = local_coords_[i];
      auto &grad_i = grads[i];
      grad_i[X] = local_i[X] * factor_yz[i];
      grad_i[Y] = local_i[Y] * factor_zx[i];
      grad_i[Z] = local_i[Z] * factor_xy[i];
    }
    grads[4][X] = grads[4][Y] = 0;
    grads[4][Z] = 0.5;
  }
  std::vector<Local> LocalToShapeGradients(
      Scalar x_local, Scalar y_local, Scalar z_local) const final {
    auto grads = std::vector<Local>(kNodes);
    LocalToShapeGradients(x_local, y_local, z_local, grads.data());
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
  Pyramid5(std::initializer_list<Global> il) {
    Element<Scalar, 3, 3>::_Build(this, il);
  }
};
// initialization of static const members:
template <std::floating_point Scalar>
const std::array<typename Pyramid5<Scalar>::Local, 5>
Pyramid5<Scalar>::local_coords_{
  Pyramid5::Local(-1, -1, -1), Pyramid5::Local(+1, -1, -1),
  Pyramid5::Local(+1, +1, -1), Pyramid5::Local(-1, +1, -1),
  Pyramid5::Local(0, 0, 1)
};
template <std::floating_point Scalar>
const std::array<std::array<int, 3>, 4>
Pyramid5<Scalar>::faces_{
  /* See http://cgns.github.io/CGNS_docs_current/sids/conv.figs/pyra_5.png for node numbering.
   */
  0, 1, 4,
  1, 2, 4,
  2, 3, 4,
  3, 0, 4
};

template <std::floating_point Scalar>
class Pyramid14;

/**
 * @brief Coordinate map on 13-node pyramidal elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Pyramid13 : public Pyramid<Scalar> {
  using Base = Pyramid<Scalar>;
  friend class Pyramid14<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  static constexpr int kNodes = 13;

 private:
  std::array<Global, kNodes> global_coords_;
  static const std::array<Local, kNodes> local_coords_;
  static const std::array<std::array<int, 6>, 4> faces_;

 public:
  int CountNodes() const final {
    return kNodes;
  }
  void SortNodesOnFace(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node) const final {
    if (face_n_node == 6) {
      int i_face = Base::GetFaceId(cell_nodes, face_nodes, face_n_node);
      for (int f = 0; f < face_n_node; ++f) {
        face_nodes[f] = cell_nodes[faces_[i_face][f]];
      }
    } else {
      assert(face_n_node == 8);
      // the face is the bottom
      face_nodes[0] = cell_nodes[0];
      face_nodes[1] = cell_nodes[3];
      face_nodes[2] = cell_nodes[2];
      face_nodes[3] = cell_nodes[1];
      face_nodes[4] = cell_nodes[8];
      face_nodes[5] = cell_nodes[7];
      face_nodes[6] = cell_nodes[6];
      face_nodes[7] = cell_nodes[5];
    }
  }

 private:
  static void LocalToNewShapeFunctions(Scalar x_local, Scalar y_local,
      Scalar z_local, Scalar *shapes) {
    auto quadratic_x = (1 - x_local * x_local);
    auto quadratic_y = (1 - y_local * y_local);
    auto quadratic_z = (1 - z_local * z_local) / 4;
    for (int i : {5, 7}) {  // local_coord_[i][X] = 0
      auto factor_y = (1 + local_coords_[i][Y] * y_local) / 2;
      auto factor_z = (1 + local_coords_[i][Z] * z_local) / 2;
      shapes[i] = quadratic_x * factor_y * (factor_z - quadratic_z);
    }
    for (int i : {6, 8}) {  // local_coord_[i][Y] = 0
      auto factor_x = (1 + local_coords_[i][X] * x_local) / 2;
      auto factor_z = (1 + local_coords_[i][Z] * z_local) / 2;
      shapes[i] = factor_x * quadratic_y * (factor_z - quadratic_z);
    }
    for (int i : {9, 10, 11, 12}) {  // local_coord_[i][Z] = 0
      auto factor_x = (1 + local_coords_[i][X] * x_local);
      auto factor_y = (1 + local_coords_[i][Y] * y_local);
      shapes[i] = factor_x * factor_y * quadratic_z;
    }
  }
  static void LocalToNewShapeGradients(Scalar x_local, Scalar y_local,
      Scalar z_local, Local *grads) {
    auto quadratic_x = (1 - x_local * x_local);
    auto quadratic_y = (1 - y_local * y_local);
    auto quadratic_z = (1 - z_local * z_local) / 4;
    for (int i : {5, 7}) {  // local_coord_[i][X] = 0
      auto factor_y = (1 + local_coords_[i][Y] * y_local) / 2;
      auto factor_z = (1 + local_coords_[i][Z] * z_local) / 2;
      // shapes[i] = quadratic_x * factor_y * (factor_z - quadratic_z);
      factor_z -= quadratic_z;
      grads[i][X] = (-2 * x_local) * factor_y * factor_z;
      grads[i][Y] = (local_coords_[i][Y] / 2) * quadratic_x * factor_z;
      auto grad_z = (local_coords_[i][Z] + z_local) / 2;
      grads[i][Z] = quadratic_x * factor_y * grad_z;
    }
    for (int i : {6, 8}) {  // local_coord_[i][Y] = 0
      auto factor_x = (1 + local_coords_[i][X] * x_local) / 2;
      auto factor_z = (1 + local_coords_[i][Z] * z_local) / 2;
      // shapes[i] = factor_x * quadratic_y * (factor_z - quadratic_z);
      factor_z -= quadratic_z;
      grads[i][Y] = (-2 * y_local) * factor_x * factor_z;
      grads[i][X] = (local_coords_[i][X] / 2) * quadratic_y * factor_z;
      auto grad_z = (local_coords_[i][Z] + z_local) / 2;
      grads[i][Z] = factor_x * quadratic_y * grad_z;
    }
    for (int i : {9, 10, 11, 12}) {  // local_coord_[i][Z] = 0
      auto factor_x = (1 + local_coords_[i][X] * x_local);
      auto factor_y = (1 + local_coords_[i][Y] * y_local);
      // shapes[i] = factor_x * factor_y * quadratic_z;
      grads[i][X] = local_coords_[i][X] * factor_y * quadratic_z;
      grads[i][Y] = local_coords_[i][Y] * factor_x * quadratic_z;
      grads[i][Z] = factor_x * factor_y * (-z_local / 2);
    }
  }

 public:
  static void LocalToShapeFunctions(Scalar x_local, Scalar y_local,
      Scalar z_local, Scalar *shapes) {
    Pyramid5<Scalar>::LocalToShapeFunctions(
        x_local, y_local, z_local, shapes);
    LocalToNewShapeFunctions(x_local, y_local, z_local, shapes);
    for (int b = 5; b < 13; ++b) {
      Scalar x_b = local_coords_[b][X];
      Scalar y_b = local_coords_[b][Y];
      Scalar z_b = local_coords_[b][Z];
      Scalar old_shapes_on_new_nodes[5];
      Pyramid5<Scalar>::LocalToShapeFunctions(
          x_b, y_b, z_b, old_shapes_on_new_nodes);
      for (int a = 0; a < 5; ++a) {
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
    Pyramid5<Scalar>::LocalToShapeGradients(
        x_local, y_local, z_local, grads);
    LocalToNewShapeGradients(x_local, y_local, z_local, grads);
    for (int b = 5; b < 13; ++b) {
      Scalar x_b = local_coords_[b][X];
      Scalar y_b = local_coords_[b][Y];
      Scalar z_b = local_coords_[b][Z];
      Scalar old_shapes_on_new_nodes[5];
      Pyramid5<Scalar>::LocalToShapeFunctions(
          x_b, y_b, z_b, old_shapes_on_new_nodes);
      for (int a = 0; a < 5; ++a) {
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
  Pyramid13(std::initializer_list<Global> il) {
    Element<Scalar, 3, 3>::_Build(this, il);
  }
};
// initialization of static const members:
template <std::floating_point Scalar>
const std::array<typename Pyramid13<Scalar>::Local, 13>
Pyramid13<Scalar>::local_coords_{
  Pyramid13::Local(-1, -1, -1), Pyramid13::Local(+1, -1, -1),
  Pyramid13::Local(+1, +1, -1), Pyramid13::Local(-1, +1, -1),
  Pyramid13::Local(0, 0, 1),
  Pyramid13::Local(0, -1, -1), Pyramid13::Local(+1, 0, -1),
  Pyramid13::Local(0, +1, -1), Pyramid13::Local(-1, 0, -1),
  Pyramid13::Local(-1, -1, 0), Pyramid13::Local(+1, -1, 0),
  Pyramid13::Local(+1, +1, 0), Pyramid13::Local(-1, +1, 0),
};
template <std::floating_point Scalar>
const std::array<std::array<int, 6>, 4>
Pyramid13<Scalar>::faces_{
  /* See http://cgns.github.io/CGNS_docs_current/sids/conv.figs/pyra_13.png for node numbering.
   */
  0, 1, 4, 5, 10, 9,
  1, 2, 4, 6, 11, 10,
  2, 3, 4, 7, 12, 11,
  3, 0, 4, 8, 9, 12,
};

/**
 * @brief Coordinate map on 14-node pyramidal elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Pyramid14 : public Pyramid<Scalar> {
  using Base = Pyramid<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  static constexpr int kNodes = 14;

 private:
  std::array<Global, kNodes> global_coords_;
  static const std::array<Local, kNodes> local_coords_;

 public:
  int CountNodes() const final {
    return kNodes;
  }
  void SortNodesOnFace(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node) const final {
    if (face_n_node == 6) {
      int i_face = Base::GetFaceId(cell_nodes, face_nodes, face_n_node);
      for (int f = 0; f < face_n_node; ++f) {
        face_nodes[f] = cell_nodes[Pyramid13<Scalar>::faces_[i_face][f]];
      }
    } else {
      assert(face_n_node == 9);
      // the face is the bottom
      face_nodes[0] = cell_nodes[0];
      face_nodes[1] = cell_nodes[3];
      face_nodes[2] = cell_nodes[2];
      face_nodes[3] = cell_nodes[1];
      face_nodes[4] = cell_nodes[8];
      face_nodes[5] = cell_nodes[7];
      face_nodes[6] = cell_nodes[6];
      face_nodes[7] = cell_nodes[5];
      face_nodes[8] = cell_nodes[13];
    }
  }

 private:
  static void LocalToNewShapeFunctions(Scalar x_local, Scalar y_local,
      Scalar z_local, Scalar *shapes) {
    auto quadratic_x = (1 - x_local * x_local);
    auto quadratic_y = (1 - y_local * y_local);
    auto factor_z = (1 - z_local) / 2;
    shapes[13] = quadratic_x * quadratic_y * factor_z;
  }
  static void LocalToNewShapeGradients(Scalar x_local, Scalar y_local,
      Scalar z_local, Local *grads) {
    auto quadratic_x = (1 - x_local * x_local);
    auto quadratic_y = (1 - y_local * y_local);
    auto factor_z = (1 - z_local)/* \divide 2 */;
    grads[13][X] = (/* 2 \times */-x_local) * quadratic_y * factor_z;
    grads[13][Y] = (/* 2 \times */-y_local) * quadratic_x * factor_z;
    grads[13][Z] = quadratic_x * quadratic_y * (-0.5);
  }

 public:
  static void LocalToShapeFunctions(Scalar x_local, Scalar y_local,
      Scalar z_local, Scalar *shapes) {
    Pyramid13<Scalar>::LocalToShapeFunctions(
        x_local, y_local, z_local, shapes);
    LocalToNewShapeFunctions(x_local, y_local, z_local, shapes);
    for (int b = 13; b < 14; ++b) {
      Scalar x_b = local_coords_[b][X];
      Scalar y_b = local_coords_[b][Y];
      Scalar z_b = local_coords_[b][Z];
      Scalar old_shapes_on_new_nodes[13];
      Pyramid13<Scalar>::LocalToShapeFunctions(
          x_b, y_b, z_b, old_shapes_on_new_nodes);
      for (int a = 0; a < 13; ++a) {
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
    Pyramid13<Scalar>::LocalToShapeGradients(
        x_local, y_local, z_local, grads);
    LocalToNewShapeGradients(x_local, y_local, z_local, grads);
    for (int b = 13; b < 14; ++b) {
      Scalar x_b = local_coords_[b][X];
      Scalar y_b = local_coords_[b][Y];
      Scalar z_b = local_coords_[b][Z];
      Scalar old_shapes_on_new_nodes[13];
      Pyramid13<Scalar>::LocalToShapeFunctions(
          x_b, y_b, z_b, old_shapes_on_new_nodes);
      for (int a = 0; a < 13; ++a) {
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
  Pyramid14(std::initializer_list<Global> il) {
    Element<Scalar, 3, 3>::_Build(this, il);
  }
};
// initialization of static const members:
template <std::floating_point Scalar>
const std::array<typename Pyramid14<Scalar>::Local, 14>
Pyramid14<Scalar>::local_coords_{
  Pyramid14::Local(-1, -1, -1), Pyramid14::Local(+1, -1, -1),
  Pyramid14::Local(+1, +1, -1), Pyramid14::Local(-1, +1, -1),
  Pyramid14::Local(0, 0, 1),
  Pyramid14::Local(0, -1, -1), Pyramid14::Local(+1, 0, -1),
  Pyramid14::Local(0, +1, -1), Pyramid14::Local(-1, 0, -1),
  Pyramid14::Local(-1, -1, 0), Pyramid14::Local(+1, -1, 0),
  Pyramid14::Local(+1, +1, 0), Pyramid14::Local(-1, +1, 0),
  Pyramid14::Local(0, 0, -1),
};

}  // namespace geometry
}  // namespace mini

#endif  // MINI_GEOMETRY_PYRAMID_HPP_
