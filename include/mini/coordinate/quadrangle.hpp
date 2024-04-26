//  Copyright 2023 PEI Weicheng
#ifndef MINI_GEOMETRY_QUADRANGLE_HPP_
#define MINI_GEOMETRY_QUADRANGLE_HPP_

#include <concepts>

#include <cassert>

#include <array>
#include <initializer_list>
#include <vector>

#include "mini/geometry/element.hpp"
#include "mini/geometry/face.hpp"

namespace mini {
namespace geometry {

/**
 * @brief Abstract coordinate map on quadrilateral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar, int kPhysDim>
class Quadrangle : public Face<Scalar, kPhysDim> {
  using Base = Face<Scalar, kPhysDim>;

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
    Scalar a = 0;
    center_ = this->LocalToGlobal(a, a);
  }
};

/**
 * @brief Coordinate map on 4-node quadrilateral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar, int kPhysDim>
class Quadrangle4 : public Quadrangle<Scalar, kPhysDim> {
  using Base = Quadrangle<Scalar, kPhysDim>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  static constexpr int kNodes = 4;

 private:
  std::array<Global, kNodes> global_coords_;
  static const std::array<Local, kNodes> local_coords_;

 public:
  int CountNodes() const final {
    return kNodes;
  }

 public:
  static void LocalToShapeFunctions(Scalar x_local, Scalar y_local,
      Scalar *shapes) {
    for (int i = 0; i < kNodes; ++i) {
      auto &local_i = local_coords_[i];
      auto val = (1 + local_i[X] * x_local) * (1 + local_i[Y] * y_local);
      shapes[i] = val / 4;
    }
  }
  std::vector<Scalar> LocalToShapeFunctions(Scalar x_local, Scalar y_local)
      const final {
    auto shapes = std::vector<Scalar>(kNodes);
    LocalToShapeFunctions(x_local, y_local, shapes.data());
    return shapes;
  }
  static void LocalToShapeGradients(Scalar x_local, Scalar y_local,
      Local *grads) {
    std::array<Scalar, kNodes> factor_x, factor_y;
    for (int i = 0; i < kNodes; ++i) {
      auto &local_i = local_coords_[i];
      factor_x[i] = (1 + local_i[X] * x_local) / 4;
      factor_y[i] = (1 + local_i[Y] * y_local) / 4;
    }
    for (int i = 0; i < kNodes; ++i) {
      auto &local_i = local_coords_[i];
      auto &grad_i = grads[i];
      grad_i[X] = local_i[X] * factor_y[i];
      grad_i[Y] = local_i[Y] * factor_x[i];
    }
  }
  std::vector<Local> LocalToShapeGradients(Scalar x_local, Scalar y_local)
      const final {
    auto grads = std::vector<Local>(kNodes);
    LocalToShapeGradients(x_local, y_local, grads.data());
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
  Quadrangle4(
      Global const &p0, Global const &p1,
      Global const &p2, Global const &p3) {
    global_coords_[0] = p0; global_coords_[1] = p1;
    global_coords_[2] = p2; global_coords_[3] = p3;
    this->_BuildCenter();
  }

  Quadrangle4(std::initializer_list<Global> il) {
    Element<Scalar, kPhysDim, 2>::_Build(this, il);
  }
};
// initialization of static const members:
template <std::floating_point Scalar, int kPhysDim>
const std::array<typename Quadrangle4<Scalar, kPhysDim>::Local, 4>
Quadrangle4<Scalar, kPhysDim>::local_coords_{
  Quadrangle4::Local(-1, -1), Quadrangle4::Local(+1, -1),
  Quadrangle4::Local(+1, +1), Quadrangle4::Local(-1, +1)
};

/**
 * @brief Coordinate map on 8-node quadrilateral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar, int kPhysDim>
class Quadrangle8 : public Quadrangle<Scalar, kPhysDim> {
  using Base = Quadrangle<Scalar, kPhysDim>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  static constexpr int kNodes = 8;

 private:
  std::array<Global, kNodes> global_coords_;
  static const std::array<Local, kNodes> local_coords_;

 public:
  int CountNodes() const final {
    return kNodes;
  }

 private:
  static void LocalToNewShapeFunctions(Scalar x_local, Scalar y_local,
      Scalar *shapes) {
    Scalar factor_x = (1 - x_local * x_local) / 2;
    shapes[4] = factor_x * (1 + local_coords_[4][Y] * y_local);
    shapes[6] = factor_x * (1 + local_coords_[6][Y] * y_local);
    Scalar factor_y = (1 - y_local * y_local) / 2;
    shapes[5] = (1 + local_coords_[5][X] * x_local) * factor_y;
    shapes[7] = (1 + local_coords_[7][X] * x_local) * factor_y;
  }
  static void LocalToNewShapeGradients(Scalar x_local, Scalar y_local,
      Local *grads) {
    Scalar factor_x = (1 - x_local * x_local) / 2;
    grads[4][X] = -x_local * (1 + local_coords_[4][Y] * y_local);
    grads[4][Y] = factor_x * local_coords_[4][Y];
    grads[6][X] = -x_local * (1 + local_coords_[6][Y] * y_local);
    grads[6][Y] = factor_x * local_coords_[6][Y];
    Scalar factor_y = (1 - y_local * y_local) / 2;
    grads[5][X] = factor_y * local_coords_[5][X];
    grads[5][Y] = -y_local * (1 + local_coords_[5][X] * x_local);
    grads[7][X] = factor_y * local_coords_[7][X];
    grads[7][Y] = -y_local * (1 + local_coords_[7][X] * x_local);
  }

 public:
  static void LocalToShapeFunctions(Scalar x_local, Scalar y_local,
      Scalar *shapes) {
    Quadrangle4<Scalar, kPhysDim>::LocalToShapeFunctions(
        x_local, y_local, shapes);
    LocalToNewShapeFunctions(x_local, y_local, shapes);
    for (int b = 4; b < 8; ++b) {
      Scalar x_b = local_coords_[b][X];
      Scalar y_b = local_coords_[b][Y];
      Scalar old_shapes_on_new_nodes[4];
      Quadrangle4<Scalar, kPhysDim>::LocalToShapeFunctions(
          x_b, y_b, old_shapes_on_new_nodes);
      for (int a = 0; a < 4; ++a) {
        shapes[a] -= old_shapes_on_new_nodes[a] * shapes[b];
      }
    }
  }
  std::vector<Scalar> LocalToShapeFunctions(Scalar x_local, Scalar y_local)
      const final {
    auto shapes = std::vector<Scalar>(kNodes);
    LocalToShapeFunctions(x_local, y_local, shapes.data());
    return shapes;
  }
  static void LocalToShapeGradients(Scalar x_local, Scalar y_local,
      Local *grads) {
    Quadrangle4<Scalar, kPhysDim>::LocalToShapeGradients(
        x_local, y_local, grads);
    LocalToNewShapeGradients(x_local, y_local, grads);
    for (int b = 4; b < 8; ++b) {
      Scalar x_b = local_coords_[b][X];
      Scalar y_b = local_coords_[b][Y];
      Scalar old_shapes_on_new_nodes[4];
      Quadrangle4<Scalar, kPhysDim>::LocalToShapeFunctions(
          x_b, y_b, old_shapes_on_new_nodes);
      for (int a = 0; a < 4; ++a) {
        grads[a] -= old_shapes_on_new_nodes[a] * grads[b];
      }
    }
  }
  std::vector<Local> LocalToShapeGradients(Scalar x_local, Scalar y_local)
      const final {
    auto grads = std::vector<Local>(kNodes);
    LocalToShapeGradients(x_local, y_local, grads.data());
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
  Quadrangle8(std::initializer_list<Global> il) {
    Element<Scalar, kPhysDim, 2>::_Build(this, il);
  }
};
// initialization of static const members:
template <std::floating_point Scalar, int kPhysDim>
const std::array<typename Quadrangle8<Scalar, kPhysDim>::Local, 8>
Quadrangle8<Scalar, kPhysDim>::local_coords_{
  Quadrangle8::Local(-1, -1), Quadrangle8::Local(+1, -1),
  Quadrangle8::Local(+1, +1), Quadrangle8::Local(-1, +1),
  Quadrangle8::Local(0, -1), Quadrangle8::Local(+1, 0),
  Quadrangle8::Local(0, +1), Quadrangle8::Local(-1, 0)
};

/**
 * @brief Coordinate map on 9-node quadrilateral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar, int kPhysDim>
class Quadrangle9 : public Quadrangle<Scalar, kPhysDim> {
  using Base = Quadrangle<Scalar, kPhysDim>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  static constexpr int kNodes = 9;

 private:
  std::array<Global, kNodes> global_coords_;
  static const std::array<Local, kNodes> local_coords_;

 public:
  int CountNodes() const final {
    return kNodes;
  }

 private:
  static void LocalToNewShapeFunctions(Scalar x_local, Scalar y_local,
      Scalar *shapes) {
    Scalar factor_x = (1 - x_local * x_local);
    Scalar factor_y = (1 - y_local * y_local);
    shapes[8] = factor_x * factor_y;
  }
  static void LocalToNewShapeGradients(Scalar x_local, Scalar y_local,
      Local *grads) {
    grads[8][X] = (-2 * x_local) * (1 - y_local * y_local);
    grads[8][Y] = (1 - x_local * x_local) * (-2 * y_local);
  }

 public:
  static void LocalToShapeFunctions(Scalar x_local, Scalar y_local,
      Scalar *shapes) {
    Quadrangle8<Scalar, kPhysDim>::LocalToShapeFunctions(
        x_local, y_local, shapes);
    LocalToNewShapeFunctions(x_local, y_local, shapes);
    for (int b = 8; b < 9; ++b) {
      Scalar x_b = local_coords_[b][X];
      Scalar y_b = local_coords_[b][Y];
      Scalar old_shapes_on_new_nodes[8];
      Quadrangle8<Scalar, kPhysDim>::LocalToShapeFunctions(
          x_b, y_b, old_shapes_on_new_nodes);
      for (int a = 0; a < 8; ++a) {
        shapes[a] -= old_shapes_on_new_nodes[a] * shapes[b];
      }
    }
  }
  std::vector<Scalar> LocalToShapeFunctions(Scalar x_local, Scalar y_local)
      const final {
    auto shapes = std::vector<Scalar>(kNodes);
    LocalToShapeFunctions(x_local, y_local, shapes.data());
    return shapes;
  }
  static void LocalToShapeGradients(Scalar x_local, Scalar y_local,
      Local *grads) {
    Quadrangle8<Scalar, kPhysDim>::LocalToShapeGradients(
        x_local, y_local, grads);
    LocalToNewShapeGradients(x_local, y_local, grads);
    for (int b = 8; b < 9; ++b) {
      Scalar x_b = local_coords_[b][X];
      Scalar y_b = local_coords_[b][Y];
      Scalar old_shapes_on_new_nodes[8];
      Quadrangle8<Scalar, kPhysDim>::LocalToShapeFunctions(
          x_b, y_b, old_shapes_on_new_nodes);
      for (int a = 0; a < 8; ++a) {
        grads[a] -= old_shapes_on_new_nodes[a] * grads[b];
      }
    }
  }
  std::vector<Local> LocalToShapeGradients(Scalar x_local, Scalar y_local)
      const final {
    auto grads = std::vector<Local>(kNodes);
    LocalToShapeGradients(x_local, y_local, grads.data());
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
  Quadrangle9(
      Global const &p0, Global const &p1,
      Global const &p2, Global const &p3,
      Global const &p4, Global const &p5,
      Global const &p6, Global const &p7, Global const &p8) {
    global_coords_[0] = p0; global_coords_[1] = p1;
    global_coords_[2] = p2; global_coords_[3] = p3;
    global_coords_[4] = p4; global_coords_[5] = p5;
    global_coords_[6] = p6; global_coords_[7] = p7;
    global_coords_[8] = p8;
    this->_BuildCenter();
  }

  Quadrangle9(std::initializer_list<Global> il) {
    Element<Scalar, kPhysDim, 2>::_Build(this, il);
  }
};
// initialization of static const members:
template <std::floating_point Scalar, int kPhysDim>
const std::array<typename Quadrangle9<Scalar, kPhysDim>::Local, 9>
Quadrangle9<Scalar, kPhysDim>::local_coords_{
  Quadrangle9::Local(-1, -1), Quadrangle9::Local(+1, -1),
  Quadrangle9::Local(+1, +1), Quadrangle9::Local(-1, +1),
  Quadrangle9::Local(0, -1), Quadrangle9::Local(+1, 0),
  Quadrangle9::Local(0, +1), Quadrangle9::Local(-1, 0), Quadrangle9::Local(0, 0)
};

}  // namespace geometry
}  // namespace mini

#endif  // MINI_GEOMETRY_QUADRANGLE_HPP_
