//  Copyright 2023 PEI Weicheng
#ifndef MINI_GEOMETRY_TRIANGLE_HPP_
#define MINI_GEOMETRY_TRIANGLE_HPP_

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
 * @brief Abstract coordinate map on triangular elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam kPhysDim  Dimension of the underlying physical space.
 */
template <std::floating_point Scalar, int kPhysDim>
class Triangle : public Face<Scalar, kPhysDim> {
  using Base = Face<Scalar, kPhysDim>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  int CountCorners() const final {
    return 3;
  }
  const Global &center() const final {
    return center_;
  }

 protected:
  Global center_;
  void _BuildCenter() final {
    Scalar a = 1.0 / 3;
    center_ = this->LocalToGlobal(a, a);
  }
};

/**
 * @brief Coordinate map on 3-node triangular elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam kPhysDim  Dimension of the underlying physical space.
 */
template <std::floating_point Scalar, int kPhysDim>
class Triangle3 : public Triangle<Scalar, kPhysDim> {
  using Base = Triangle<Scalar, kPhysDim>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  static constexpr int kNodes = 3;

 private:
  std::array<Global, kNodes> global_coords_;
  static const std::array<Local, kNodes> local_coords_;

 public:
  int CountNodes() const final {
    return kNodes;
  }

 public:
  std::vector<Scalar> LocalToShapeFunctions(Scalar x_local, Scalar y_local)
      const final {
    return {
      x_local, y_local, 1.0 - x_local - y_local
    };
  }
  std::vector<Local> LocalToShapeGradients(Scalar x_local, Scalar y_local)
      const final {
    return {
      Local(1, 0), Local(0, 1), Local(-1, -1)
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
  Triangle3(std::initializer_list<Global> il) {
    Element<Scalar, kPhysDim, 2>::_Build(this, il);
  }
};
// initialization of static const members:
template <std::floating_point Scalar, int kPhysDim>
const std::array<typename Triangle3<Scalar, kPhysDim>::Local, 3>
Triangle3<Scalar, kPhysDim>::local_coords_{
  Triangle3::Local(1, 0), Triangle3::Local(0, 1),
  Triangle3::Local(0, 0)
};

/**
 * @brief Coordinate map on 6-node triangular elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam kPhysDim  Dimension of the underlying physical space.
 */
template <std::floating_point Scalar, int kPhysDim>
class Triangle6 : public Triangle<Scalar, kPhysDim> {
  using Base = Triangle<Scalar, kPhysDim>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  static constexpr int kNodes = 6;

 private:
  std::array<Global, kNodes> global_coords_;
  static const std::array<Local, kNodes> local_coords_;

 public:
  int CountNodes() const final {
    return kNodes;
  }

 public:
  std::vector<Scalar> LocalToShapeFunctions(Scalar x_local, Scalar y_local)
      const final {
    auto shapes = std::vector<Scalar>(kNodes);
    auto z_local = 1.0 - x_local - y_local;
    shapes[0] = x_local * (x_local - 0.5) * 2;
    shapes[1] = y_local * (y_local - 0.5) * 2;
    shapes[2] = z_local * (z_local - 0.5) * 2;
    shapes[3] = x_local * y_local * 4;
    shapes[4] = y_local * z_local * 4;
    shapes[5] = z_local * x_local * 4;
    return shapes;
  }
  std::vector<Local> LocalToShapeGradients(Scalar x_local, Scalar y_local)
      const final {
    auto grads = std::vector<Local>(kNodes);
    auto factor_x = 4 * x_local;
    auto factor_y = 4 * y_local;
    auto factor_z = 4 - factor_x - factor_y;
    // shapes[0] = x_local * (x_local - 0.5) * 2;
    grads[0][X] = factor_x - 1;
    grads[0][Y] = 0;
    // shapes[1] = y_local * (y_local - 0.5) * 2;
    grads[1][X] = 0;
    grads[1][Y] = factor_y - 1;
    // shapes[2] = z_local * (z_local - 0.5) * 2;
    grads[2][X] = grads[2][Y] = 1 - factor_z;
    // shapes[3] = x_local * y_local * 4;
    grads[3][X] = factor_y;
    grads[3][Y] = factor_x;
    // shapes[4] = y_local * (1 - x_local - y_local) * 4;
    grads[4][X] = -factor_y;
    grads[4][Y] = -factor_y + factor_z;
    // shapes[5] = (1 - x_local - y_local) * x_local * 4;
    grads[5][X] = -factor_x + factor_z;
    grads[5][Y] = -factor_x;
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
  Triangle6(std::initializer_list<Global> il) {
    Element<Scalar, kPhysDim, 2>::_Build(this, il);
  }
};
// initialization of static const members:
template <std::floating_point Scalar, int kPhysDim>
const std::array<typename Triangle6<Scalar, kPhysDim>::Local, 6>
Triangle6<Scalar, kPhysDim>::local_coords_{
  Triangle6::Local(1, 0), Triangle6::Local(0, 1),
  Triangle6::Local(0, 0), Triangle6::Local(0.5, 0.5),
  Triangle6::Local(0, 0.5), Triangle6::Local(0.5, 0)
};

}  // namespace geometry
}  // namespace mini

#endif  // MINI_GEOMETRY_TRIANGLE_HPP_
