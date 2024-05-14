//  Copyright 2023 PEI Weicheng
#ifndef MINI_COORDINATE_FACE_HPP_
#define MINI_COORDINATE_FACE_HPP_

#include <concepts>

#include <algorithm>
#include <array>
#include <memory>
#include <type_traits>
#include <vector>

#include "mini/coordinate/element.hpp"

namespace mini {
namespace coordinate {

/**
 * @brief Abstract coordinate map on surface elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam kPhysDim  Dimension of the physical space.
 */
template <std::floating_point Scalar, int kPhysDim>
class Face : public Element<Scalar, kPhysDim, 2> {
  static constexpr int D = kPhysDim;
  static_assert(D == 2 || D == 3);

 public:
  using Base = Element<Scalar, kPhysDim, 2>;
  using Real = typename Base::Real;
  using Local = typename Base::Local;
  using Global = typename Base::Global;
  using Jacobian = typename Base::Jacobian;
  using Frame = std::conditional_t<D == 3, std::array<Global, 3>, int>;

  virtual std::unique_ptr<Face> Clone() const = 0;

  virtual std::vector<Scalar> LocalToShapeFunctions(Scalar, Scalar) const = 0;
  virtual std::vector<Local> LocalToShapeGradients(Scalar, Scalar) const = 0;

  std::vector<Scalar> LocalToShapeFunctions(const Local &xy)
      const final {
    return LocalToShapeFunctions(xy[X], xy[Y]);
  }
  std::vector<Local> LocalToShapeGradients(const Local &xy)
      const final {
    return LocalToShapeGradients(xy[X], xy[Y]);
  }
  Global LocalToNormalVector(Scalar x_local, Scalar y_local) const {
    auto jacobian = LocalToJacobian(x_local, y_local);
    return jacobian.row(X).cross(jacobian.row(Y)).normalized();
  }
  Frame LocalToNormalFrame(Scalar x_local, Scalar y_local) const {
    Frame frame;
    auto &normal = frame[X], &tangent = frame[Y], &bitangent = frame[Z];
    auto jacobian = LocalToJacobian(x_local, y_local);
    normal = jacobian.row(X).cross(jacobian.row(Y)).normalized();
    tangent = jacobian.row(X).normalized();
    bitangent = normal.cross(tangent);
    return frame;
  }
  Frame LocalToNormalFrame(const Local &xy) const {
    return LocalToNormalFrame(xy[X], xy[Y]);
  }

  Global LocalToGlobal(Scalar x_local, Scalar y_local) const {
    auto shapes = LocalToShapeFunctions(x_local, y_local);
    Global sum = this->GetGlobal(0) * shapes[0];
    for (int i = 1, n = this->CountNodes(); i < n; ++i) {
      sum += this->GetGlobal(i) * shapes[i];
    }
    return sum;
  }
  Global LocalToGlobal(const Local &xy) const final {
    return LocalToGlobal(xy[X], xy[Y]);
  }

  Jacobian LocalToJacobian(Scalar x_local, Scalar y_local) const {
    auto shapes = LocalToShapeGradients(x_local, y_local);
    Jacobian sum = shapes[0] * this->GetGlobal(0).transpose();
    for (int i = 1, n = this->CountNodes(); i < n; ++i) {
      sum += shapes[i] * this->GetGlobal(i).transpose();
    }
    return sum;
  }
  Jacobian LocalToJacobian(const Local &xy) const final {
    return LocalToJacobian(xy[X], xy[Y]);
  }

  Local GlobalToLocal(const Global &xy_global,
      const Local &hint = Local(0, 0)) const {
    return this->Base::GlobalToLocal(xy_global, hint);
  }

  Local GlobalToLocal(Scalar x_global, Scalar y_global,
      const Local &hint = Local(0, 0)) const {
    Global xy_global = {x_global, y_global};
    return GlobalToLocal(xy_global, hint);
  }
};

}  // namespace coordinate
}  // namespace mini

#endif  // MINI_COORDINATE_FACE_HPP_
