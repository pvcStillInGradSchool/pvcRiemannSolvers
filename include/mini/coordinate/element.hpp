//  Copyright 2023 PEI Weicheng
#ifndef MINI_COORDINATE_ELEMENT_HPP_
#define MINI_COORDINATE_ELEMENT_HPP_

#include <concepts>

#include <cassert>
#include <cmath>
#include <numeric>

#include <initializer_list>
#include <iostream>
#include <vector>

#include "mini/algebra/eigen.hpp"
#include "mini/algebra/root.hpp"
#include "mini/constant/index.hpp"

namespace mini {

/**
 * @brief Maps between physical (global) coordinates and parametric (local) coordinates on regular elements.
 * 
 */
namespace coordinate {

using namespace mini::constant::index;

/**
 * @brief Abstract coordinate map on surface elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam kPhysDim  Dimension of the underlying physical space.
 * @tparam kCellDim  Dimension of the element as a manifold.
 */
template <std::floating_point Scalar, int kPhysDim, int kCellDim>
class Element {
 public:
  using Real = Scalar;
  using Local = algebra::Matrix<Scalar, kCellDim, 1>;
  using Global = algebra::Matrix<Scalar, kPhysDim, 1>;

  /**
   * @brief The type of (geometric) Jacobian matrix, which is defined as \f$ \mathbf{J}=\begin{bmatrix}\partial_{\xi}\\\partial_{\eta}\\\partial_{\zeta}\end{bmatrix}\begin{bmatrix}x & y & z\end{bmatrix} \f$.
   * 
   */
  using Jacobian = algebra::Matrix<Scalar, kCellDim, kPhysDim>;

  static constexpr int CellDim() { return kCellDim; }
  static constexpr int PhysDim() { return kPhysDim; }

  virtual ~Element() noexcept = default;
  virtual std::vector<Scalar> LocalToShapeFunctions(const Local &) const = 0;

  /**
   * @brief \f$ \begin{bmatrix}\partial_{\xi}\\ \partial_{\eta}\\ \cdots \end{bmatrix}\begin{bmatrix}\phi_{1} & \phi_{2} & \cdots\end{bmatrix}=\begin{bmatrix}\partial_{\xi}\phi_{1} & \partial_{\xi}\phi_{2} & \cdots\\ \partial_{\eta}\phi_{1} & \partial_{\eta}\phi_{2} & \cdots\\ \cdots & \cdots & \cdots \end{bmatrix} \f$
   * 
   * @return std::vector<Local> 
   */
  virtual std::vector<Local> LocalToShapeGradients(const Local &) const = 0;
  virtual Global LocalToGlobal(const Local &) const = 0;
  virtual Jacobian LocalToJacobian(const Local &) const = 0;
  virtual int CountCorners() const = 0;
  virtual int CountNodes() const = 0;
  virtual const Local &GetLocal(int i) const = 0;
  virtual const Global &GetGlobal(int i) const = 0;
  virtual const Global &center() const = 0;

 protected:
  virtual void _BuildCenter() = 0;
  Global &_GetGlobal(int i) {
    const Global &global
        = const_cast<const Element *>(this)->GetGlobal(i);
    return const_cast<Global &>(global);
  }

  static void _Build(Element *element,
      std::initializer_list<Global> il) {
    assert(il.size() == element->CountNodes());
    auto p = il.begin();
    for (int i = 0, n = element->CountNodes(); i < n; ++i) {
      element->_GetGlobal(i) = p[i];
    }
    element->_BuildCenter();
  }

 public:
  Local GlobalToLocal(const Global &global,
      const Local &hint = Local::Zero()) const requires(kCellDim == kPhysDim) {
    auto func = [this, &global](Local const &local) {
      auto res = LocalToGlobal(local);
      return res -= global;
    };
    auto jac = [this](Local const &local) {
      return LocalToJacobian(local);
    };
    Local local;
    try {
      local = mini::algebra::root::Newton(hint, func, jac);
    } catch(std::runtime_error &e) {
      std::cerr << "global = " << global.transpose() << "\n";
      std::cerr << "global_coords =" << "\n";
      for (int i = 0; i < this->CountNodes(); ++i) {
        std::cerr << this->GetGlobal(i).transpose() << "\n";
      }
      std::cerr << std::endl;
      throw e;
    }
    return local;
  }
};

}  // namespace coordinate
}  // namespace mini

#endif  // MINI_COORDINATE_ELEMENT_HPP_
