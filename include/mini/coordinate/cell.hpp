//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_COORDINATE_CELL_HPP_
#define MINI_COORDINATE_CELL_HPP_

#include <concepts>

#include <algorithm>
#include <memory>
#include <type_traits>
#include <vector>

#include "mini/coordinate/element.hpp"

namespace mini {
namespace coordinate {

/**
 * @brief Abstract coordinate map on volume elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Cell : public Element<Scalar, 3, 3> {
 public:
  using Base = Element<Scalar, 3, 3>;
  using Real = typename Base::Real;
  using Local = typename Base::Local;
  using Global = typename Base::Global;
  using Jacobian = typename Base::Jacobian;

  virtual std::unique_ptr<Cell>Clone() const = 0;

  /**
   * @brief The type of (geometric) Hessian matrix of a scalar function \f$ f \f$, which is defined as \f$ \begin{bmatrix}\partial_{\xi}\partial_{\xi}f & \partial_{\xi}\partial_{\eta}f & \partial_{\xi}\partial_{\zeta}f\\ \partial_{\eta}\partial_{\xi}f & \partial_{\eta}\partial_{\eta}f & \partial_{\eta}\partial_{\zeta}f\\ \partial_{\zeta}\partial_{\xi}f & \partial_{\zeta}\partial_{\eta}f & \partial_{\zeta}\partial_{\zeta}f \end{bmatrix} \f$.
   * 
   * Since Hessian matrices are symmetric, only the upper part are stored.
   */
  using Hessian = algebra::Vector<Scalar, 6>;

  /**
   * @brief The type of (geometric) 3rd-order derivatives of a scalar function \f$ f \f$.
   * 
   * Since partial derivatives of smooth functions are permutable, only the upper part are stored.
   */
  using Tensor3 = algebra::Vector<Scalar, 10>;

  virtual std::vector<Scalar> LocalToShapeFunctions(Scalar, Scalar, Scalar)
      const = 0;
  virtual std::vector<Local> LocalToShapeGradients(Scalar, Scalar, Scalar)
      const = 0;
  virtual std::vector<Hessian> LocalToShapeHessians(Local const &)
      const {
    return {};
  }
  virtual std::vector<Tensor3> LocalToShape3rdOrderDerivatives(Local const &)
      const {
    return {};
  }

  std::vector<Scalar> LocalToShapeFunctions(const Local &xyz)
      const final {
    return LocalToShapeFunctions(xyz[X], xyz[Y], xyz[Z]);
  }
  std::vector<Local> LocalToShapeGradients(const Local &xyz)
      const final {
    return LocalToShapeGradients(xyz[X], xyz[Y], xyz[Z]);
  }

  Global LocalToGlobal(Scalar x_local, Scalar y_local, Scalar z_local)
      const {
    auto shapes = LocalToShapeFunctions(x_local, y_local, z_local);
    Global sum = this->GetGlobal(0) * shapes[0];
    for (int i = 1, n = this->CountNodes(); i < n; ++i) {
      sum += this->GetGlobal(i) * shapes[i];
    }
    return sum;
  }
  Global LocalToGlobal(const Local &xyz) const final {
    return LocalToGlobal(xyz[X], xyz[Y], xyz[Z]);
  }

  Jacobian LocalToJacobian(Scalar x_local, Scalar y_local, Scalar z_local)
      const {
    auto shapes = LocalToShapeGradients(x_local, y_local, z_local);
    Jacobian sum = shapes[0] * this->GetGlobal(0).transpose();
    for (int i = 1, n = this->CountNodes(); i < n; ++i) {
      sum += shapes[i] * this->GetGlobal(i).transpose();
    }
    return sum;
  }
  Jacobian LocalToJacobian(const Local &xyz) const final {
    return LocalToJacobian(xyz[X], xyz[Y], xyz[Z]);
  }
  /**
   * @brief \f$ \frac{\partial}{\partial\xi}\mathbf{J} = \frac{\partial}{\partial\xi}\begin{bmatrix} \partial_\xi x & \partial_\xi y & \partial_\xi z \\ \partial_\eta x & \partial_\eta y & \partial_\eta z \\ \partial_\zeta x & \partial_\zeta y & \partial_\zeta z \\ \end{bmatrix} \f$, in which \f$ \mathbf{J} \f$ is returned by `Element::LocalToJacobian`.
   * 
   * @param xyz 
   * @return algebra::Vector<Jacobian, 3> 
   */
  algebra::Vector<Jacobian, 3> LocalToJacobianGradient(Local const &xyz)
      const {
    auto hessians = LocalToShapeHessians(xyz);
    algebra::Vector<Jacobian, 3> grad;
    grad[X].setZero(); grad[Y].setZero(); grad[Z].setZero();
    for (int i = 0, n = this->CountNodes(); i < n; ++i) {
      auto &xyz = this->GetGlobal(i);
      auto &hessian = hessians[i];
      grad[X].row(X) += xyz * hessian[XX];
      grad[X].row(Y) += xyz * hessian[XY];
      grad[X].row(Z) += xyz * hessian[XZ];
      grad[Y].row(Y) += xyz * hessian[YY];
      grad[Y].row(Z) += xyz * hessian[YZ];
      grad[Z].row(Z) += xyz * hessian[ZZ];
    }
    grad[Y].row(X) = grad[X].row(Y);
    grad[Z].row(X) = grad[X].row(Z);
    grad[Z].row(Y) = grad[Y].row(Z);
    return grad;
  }
  algebra::Vector<Jacobian, 6> LocalToJacobianHessian(Local const &xyz)
      const {
    auto tensors = LocalToShape3rdOrderDerivatives(xyz);
    algebra::Vector<Jacobian, 6> hessian;
    hessian[XX].setZero(); hessian[XY].setZero(); hessian[XZ].setZero();
    hessian[YY].setZero(); hessian[YZ].setZero(); hessian[ZZ].setZero();
    for (int i = 0, n = this->CountNodes(); i < n; ++i) {
      auto &xyz = this->GetGlobal(i);
      auto &tensor = tensors[i];
      hessian[XX].row(X) += xyz * tensor[XXX];
      hessian[XX].row(Y) += xyz * tensor[XXY];
      hessian[XX].row(Z) += xyz * tensor[XXZ];
      hessian[XY].row(Y) += xyz * tensor[XYY];
      hessian[XY].row(Z) += xyz * tensor[XYZ];
      hessian[XZ].row(Z) += xyz * tensor[XZZ];
      hessian[YY].row(Y) += xyz * tensor[YYY];
      hessian[YY].row(Z) += xyz * tensor[YYZ];
      hessian[YZ].row(Z) += xyz * tensor[YZZ];
      hessian[ZZ].row(Z) += xyz * tensor[ZZZ];
    }
    hessian[XY].row(X) = hessian[XX].row(Y);
    hessian[XZ].row(X) = hessian[XX].row(Z);
    hessian[XZ].row(Y) = hessian[XY].row(Z);
    hessian[YY].row(X) = hessian[XY].row(Y);
    hessian[YZ].row(X) = hessian[XY].row(Z);
    hessian[YZ].row(Y) = hessian[YY].row(Z);
    hessian[ZZ].row(X) = hessian[XZ].row(Z);
    hessian[ZZ].row(Y) = hessian[YZ].row(Z);
    return hessian;
  }

  /**
   * @brief \f$ \frac{\partial}{\partial \xi}\det(\mathbf{J}) = \det(\mathbf{J})\,\mathopen{\mathrm{tr}}\left(\mathbf{J}^{-1} \frac{\partial}{\partial \xi}\mathbf{J}\right) \f$, in which \f$ \mathbf{J} \f$ is returned by `Element::LocalToJacobian` and \f$ \frac{\partial}{\partial \xi}\mathbf{J} \f$ is returned by `Cell::LocalToJacobianGradient`.
   * 
   * @param xyz 
   * @return Local 
   */
  Local LocalToJacobianDeterminantGradient(const Local &xyz) const {
    Local det_grad;
    auto mat_grad = LocalToJacobianGradient(xyz);
    Jacobian mat = LocalToJacobian(xyz);
    Scalar det = mat.determinant();
    Jacobian inv = mat.inverse();
    det_grad[X] = det * (inv * mat_grad[X]).trace();
    det_grad[Y] = det * (inv * mat_grad[Y]).trace();
    det_grad[Z] = det * (inv * mat_grad[Z]).trace();
    return det_grad;
  }
  /**
   * @brief \f$ \frac{\partial^2}{\partial\xi\,\partial\eta}\det(\mathbf{J})=\det(\mathbf{J})\left[\mathopen{\mathrm{tr}}\left(\frac{\partial^2\mathbf{J}}{\partial\xi\,\partial\eta}\right)+\mathopen{\mathrm{tr}}\left(\mathbf{J}^{-1}\frac{\partial\mathbf{J}}{\partial\xi}\right)\mathopen{\mathrm{tr}}\left(\mathbf{J}^{-1}\frac{\partial\mathbf{J}}{\partial\eta}\right)-\mathopen{\mathrm{tr}}\left(\mathbf{J}^{-1}\frac{\partial\mathbf{J}}{\partial\xi}\,\mathbf{J}^{-1}\frac{\partial\mathbf{J}}{\partial\eta}\right)\right] \f$
   * 
   * @param xyz 
   * @return Hessian 
   */
  Hessian LocalToJacobianDeterminantHessian(const Local &xyz) const {
    Hessian det_hess;
    auto mat_hess = LocalToJacobianHessian(xyz);
    Jacobian mat = LocalToJacobian(xyz);
    Jacobian inv = mat.inverse();
    det_hess[XX] = (inv * mat_hess[XX]).trace();
    det_hess[XY] = (inv * mat_hess[XY]).trace();
    det_hess[XZ] = (inv * mat_hess[XZ]).trace();
    det_hess[YY] = (inv * mat_hess[YY]).trace();
    det_hess[YZ] = (inv * mat_hess[YZ]).trace();
    det_hess[ZZ] = (inv * mat_hess[ZZ]).trace();
    auto mat_grad = LocalToJacobianGradient(xyz);
    mat_grad[X] = inv * mat_grad[X];
    mat_grad[Y] = inv * mat_grad[Y];
    mat_grad[Z] = inv * mat_grad[Z];
    auto trace_x = mat_grad[X].trace();
    auto trace_y = mat_grad[Y].trace();
    auto trace_z = mat_grad[Z].trace();
    det_hess[XX] += trace_x * trace_x - (mat_grad[X] * mat_grad[X]).trace();
    det_hess[XY] += trace_x * trace_y - (mat_grad[X] * mat_grad[Y]).trace();
    det_hess[XZ] += trace_x * trace_z - (mat_grad[X] * mat_grad[Z]).trace();
    det_hess[YY] += trace_y * trace_y - (mat_grad[Y] * mat_grad[Y]).trace();
    det_hess[YZ] += trace_y * trace_z - (mat_grad[Y] * mat_grad[Z]).trace();
    det_hess[ZZ] += trace_z * trace_z - (mat_grad[Z] * mat_grad[Z]).trace();
    det_hess *= mat.determinant();
    return det_hess;
  }

  Local GlobalToLocal(const Global &xyz_global,
      const Local &hint = Local(0, 0, 0)) const {
    return this->Base::GlobalToLocal(xyz_global, hint);
  }

  Local GlobalToLocal(Scalar x_global, Scalar y_global, Scalar z_global,
      const Local &hint = Local(0, 0, 0)) const {
    Global xyz_global = {x_global, y_global, z_global};
    return GlobalToLocal(xyz_global, hint);
  }

  /**
   * @brief Sort `cell_nodes` by `face_nodes`, so that the right-hand normal of the Face point out from the Cell.
   * 
   * @param cell_nodes  The node id list of a Cell.
   * @param face_nodes  The node id list of a Face.
   * @param face_n_node  Number of nodes on the Face.
   */
  virtual void SortNodesOnFace(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node) const = 0;
};

/**
 * @brief A generic wrapper of the virtual SortNodesOnFace method.
 * 
 * @tparam Scalar  Same as Cell::Scalar.
 * @tparam T  Type of integers in the 1st list.
 * @tparam U  Type of integers in the 2nd list.
 * @param cell  The Cell holding the Face.
 * @param cell_nodes  The node id list of the Cell.
 * @param face_nodes  The node id list of the Face.
 * @param face_n_node  Number of nodes on the Face.
 */
template<std::floating_point Scalar, std::integral T, std::integral U>
void SortNodesOnFace(const Cell<Scalar> &cell, const T *cell_nodes,
    U *face_nodes, int face_n_node) {
  size_t cell_nodes_copy[64], face_nodes_copy[32];
  size_t *cell_node_list, *face_node_list;
  if (sizeof(T) == sizeof(size_t)) {
    cell_node_list = reinterpret_cast<size_t *>(const_cast<T *>(cell_nodes));
  } else {
    cell_node_list = cell_nodes_copy;
    auto n_nodes = cell.CountNodes();
    std::copy_n(cell_nodes, n_nodes, cell_node_list);
  }
  if (sizeof(U) == sizeof(size_t)) {
    face_node_list = reinterpret_cast<size_t *>(const_cast<U *>(face_nodes));
  } else {
    face_node_list = face_nodes_copy;
    std::copy_n(face_nodes, face_n_node, face_node_list);
  }
  // Delegate the real work or sorting to the virtual function.
  cell.SortNodesOnFace(cell_node_list, face_node_list, face_n_node);
  if (sizeof(U) != sizeof(size_t)) {
    std::copy_n(face_node_list, face_n_node, face_nodes);
  }
}


}  // namespace coordinate
}  // namespace mini

#endif  // MINI_COORDINATE_CELL_HPP_
