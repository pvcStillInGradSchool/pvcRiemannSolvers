//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_GEOMETRY_TETRAHEDRON_HPP_
#define MINI_GEOMETRY_TETRAHEDRON_HPP_

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
 * @brief Abstract coordinate map on tetrahedral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Tetrahedron : public Cell<Scalar> {
  using Base = Cell<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  int CountCorners() const final {
    return 4;
  }
  const Global &center() const final {
    return center_;
  }

 protected:
  Global center_;
  void _BuildCenter() final {
    Scalar a = 1.0 / 4;
    center_ = this->LocalToGlobal(a, a, a);
  }
};

/**
 * @brief Coordinate map on 4-node tetrahedral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Tetrahedron4 : public Tetrahedron<Scalar> {
  using Base = Tetrahedron<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  static constexpr int kNodes = 4;
  static constexpr int kFaces = 4;

 private:
  std::array<Global, kNodes> global_coords_;
  static const std::array<Local, kNodes> local_coords_;
  static const std::array<std::array<int, 3>, kFaces> faces_;

 public:
  int CountNodes() const final {
    return kNodes;
  }
  void SortNodesOnFace(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node) const final {
    assert(3 == face_n_node);
    int cnt = 0, nid = 0, sum = 0;
    while (cnt < 3) {
      auto curr_node = cell_nodes[nid];
      for (int i = 0; i < 3; ++i) {
        if (face_nodes[i] == curr_node) {
          sum += nid;
          ++cnt;
          break;
        }
      }
      ++nid;
    }
    int i_face = sum - 3;
    for (int i = 0; i < 3; ++i) {
      face_nodes[i] = cell_nodes[faces_[i_face][i]];
    }
  }

 public:
  std::vector<Scalar> LocalToShapeFunctions(
      Scalar x_local, Scalar y_local, Scalar z_local) const final {
    return {
      x_local, y_local, z_local, 1.0 - x_local - y_local - z_local
    };
  }
  std::vector<Local> LocalToShapeGradients(
      Scalar x_local, Scalar y_local, Scalar z_local) const final {
    return {
      Local(1, 0, 0), Local(0, 1, 0), Local(0, 0, 1),
      Local(-1, -1, -1)
    };
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
  Tetrahedron4(
      Global const &p0, Global const &p1,
      Global const &p2, Global const &p3) {
    global_coords_[0] = p0; global_coords_[1] = p1;
    global_coords_[2] = p2; global_coords_[3] = p3;
    this->_BuildCenter();
  }

  Tetrahedron4(std::initializer_list<Global> il) {
    Element<Scalar, 3, 3>::_Build(this, il);
  }
};
// initialization of static const members:
template <std::floating_point Scalar>
const std::array<typename Tetrahedron4<Scalar>::Local, 4>
Tetrahedron4<Scalar>::local_coords_{
  Tetrahedron4::Local(1, 0, 0), Tetrahedron4::Local(0, 1, 0),
  Tetrahedron4::Local(0, 0, 1), Tetrahedron4::Local(0, 0, 0)
};

template <std::floating_point Scalar>
const std::array<std::array<int, 3>, 4>
Tetrahedron4<Scalar>::faces_{
  /* See http://cgns.github.io/CGNS_docs_current/sids/conv.figs/tetra_4.png for node numbering.
   * Faces can be distinguished by the sum of the three minimum node ids.
   */
  0, 2, 1/* 3 */, 0, 1, 3/* 4 */, 0, 3, 2/* 5 */, 1, 2, 3/* 6 */
};

/**
 * @brief Coordinate map on 10-node tetrahedral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Tetrahedron10 : public Tetrahedron<Scalar> {
  using Base = Tetrahedron<Scalar>;
  static constexpr int AB{4}, BC{5}, CA{6}, DA{7}, DB{8}, DC{9};

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  static constexpr int kNodes = 10;
  static constexpr int kFaces = 4;

 private:
  std::array<Global, kNodes> global_coords_;
  static const std::array<Local, kNodes> local_coords_;
  static const std::array<std::array<int, 6>, kFaces> faces_;

 public:
  int CountNodes() const final {
    return kNodes;
  }
  void SortNodesOnFace(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node) const final {
    assert(6 == face_n_node);
    int i_face;
    int cnt[4] = { 0, 0, 0, 0 };
    for (int f = 0; f < 6; ++f) {
      for (int c = 0; c < 4; ++c) {
        if (cell_nodes[c] == face_nodes[f]) {
          cnt[c] = true;
          break;
        }
      }
    }
    if (cnt[A]) {
      if (cnt[B]) {
        if (cnt[C]) {
          i_face = 0;
        } else {
          assert(cnt[D]);
          i_face = 1;
        }
      } else {
        assert(cnt[C] && cnt[D]);
        i_face = 2;
      }
    } else {
      assert(cnt[B] && cnt[C] && cnt[D]);
      i_face = 3;
    }
    for (int i = 0; i < 6; ++i) {
      face_nodes[i] = cell_nodes[faces_[i_face][i]];
    }
  }

 public:
  std::vector<Scalar> LocalToShapeFunctions(
      Scalar a, Scalar b, Scalar c) const final {
    auto shapes = std::vector<Scalar>(kNodes);
    auto d = 1.0 - a - b - c;
    shapes[A] = a * (a - 0.5) * 2;
    shapes[B] = b * (b - 0.5) * 2;
    shapes[C] = c * (c - 0.5) * 2;
    shapes[D] = d * (d - 0.5) * 2;
    a *= 4;
    shapes[AB] = b * a/* 4a */;
    b *= 4;
    shapes[BC] = c * b/* 4b */;
    shapes[CA] = c * a/* 4a */;
    shapes[DA] = d * a/* 4a */;
    shapes[DB] = d * b/* 4b */;
    shapes[DC] = d * c * 4;
    return shapes;
  }
  std::vector<Local> LocalToShapeGradients(
      Scalar a, Scalar b, Scalar c) const final {
    auto grads = std::vector<Local>(kNodes);
    auto factor_a = 4 * a;
    auto factor_b = 4 * b;
    auto factor_c = 4 * c;
    auto factor_d = 4 - factor_a - factor_b - factor_c;
    // shapes[A] = a * (a - 0.5) * 2;
    grads[A][A] = factor_a - 1;
    grads[A][B] = 0;
    grads[A][C] = 0;
    // shapes[B] = b * (b - 0.5) * 2;
    grads[B][A] = 0;
    grads[B][B] = factor_b - 1;
    grads[B][C] = 0;
    // shapes[C] = c * (c - 0.5) * 2;
    grads[C][A] = 0;
    grads[C][B] = 0;
    grads[C][C] = factor_c - 1;
    // shapes[D] = d * (d - 0.5) * 2 = (1 - a - b - c) * (1/2 - a - b - c) * 2;
    grads[D][A] = grads[D][B] = grads[D][C] = 1 - factor_d;
    // shapes[AB] = a * b * 4;
    grads[AB][A] = factor_b;
    grads[AB][B] = factor_a;
    grads[AB][C] = 0;
    // shapes[BC] = b * c * 4;
    grads[BC][A] = 0;
    grads[BC][B] = factor_c;
    grads[BC][C] = factor_b;
    // shapes[CA] = c * a * 4;
    grads[CA][A] = factor_c;
    grads[CA][B] = 0;
    grads[CA][C] = factor_a;
    // shapes[DA] = d * a * 4 = (1 - a - b - c) * a * 4;
    grads[DA][A] = factor_d - factor_a;
    grads[DA][B] = grads[DA][C] = -factor_a;
    // shapes[DB] = d * b * 4 = (1 - a - b - c) * b * 4;
    grads[DB][B] = factor_d - factor_b;
    grads[DB][A] = grads[DB][C] = -factor_b;
    // shapes[DC] = d * c * 4 = (1 - a - b - c) * c * 4;
    grads[DC][C] = factor_d - factor_c;
    grads[DC][A] = grads[DC][B] = -factor_c;
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
  Tetrahedron10(std::initializer_list<Global> il) {
    Element<Scalar, 3, 3>::_Build(this, il);
  }
};
// initialization of static const members:
template <std::floating_point Scalar>
const std::array<typename Tetrahedron10<Scalar>::Local, 10>
Tetrahedron10<Scalar>::local_coords_{
  /* A */Tetrahedron10::Local(1, 0, 0),
  /* B */Tetrahedron10::Local(0, 1, 0),
  /* C */Tetrahedron10::Local(0, 0, 1),
  /* D */Tetrahedron10::Local(0, 0, 0),
  /* (A + B) / 2 */Tetrahedron10::Local(0.5, 0.5, 0),
  /* (B + C) / 2 */Tetrahedron10::Local(0, 0.5, 0.5),
  /* (C + A) / 2 */Tetrahedron10::Local(0.5, 0, 0.5),
  /* (A + D) / 2 */Tetrahedron10::Local(0.5, 0, 0),
  /* (B + D) / 2 */Tetrahedron10::Local(0, 0.5, 0),
  /* (C + D) / 2 */Tetrahedron10::Local(0, 0, 0.5)
};

template <std::floating_point Scalar>
const std::array<std::array<int, 6>, 4>
Tetrahedron10<Scalar>::faces_{
  /* See http://cgns.github.io/CGNS_docs_current/sids/conv.figs/tetra_10.png for node numbering.
   */
  /* A C B */0, 2, 1, 6, 5, 4, /* A B D */0, 1, 3, 4, 8, 7,
  /* A D C */0, 3, 2, 7, 9, 6, /* B C D */1, 2, 3, 5, 9, 8
};

}  // namespace geometry
}  // namespace mini

#endif  // MINI_GEOMETRY_TETRAHEDRON_HPP_
