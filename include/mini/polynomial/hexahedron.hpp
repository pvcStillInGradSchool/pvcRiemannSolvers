//  Copyright 2023 PEI Weicheng
#ifndef MINI_POLYNOMIAL_HEXAHEDRON_HPP_
#define MINI_POLYNOMIAL_HEXAHEDRON_HPP_

#include <concepts>

#include <cmath>
#include <cstring>

#include <algorithm>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "mini/algebra/eigen.hpp"
#include "mini/gauss/cell.hpp"
#include "mini/gauss/lobatto.hpp"
#include "mini/gauss/hexahedron.hpp"
#include "mini/geometry/element.hpp"
#include "mini/basis/lagrange.hpp"
#include "mini/constant/index.hpp"

namespace mini {
namespace polynomial {

using namespace mini::constant::index;

/**
 * @brief A vector-valued function interpolated on an given basis::lagrange::Hexahedron basis.
 * 
 * The interpolation nodes are collocated with quadrature points.
 * 
 * @tparam Gx  The quadrature rule in the 1st dimension.
 * @tparam Gy  The quadrature rule in the 2nd dimension.
 * @tparam Gz  The quadrature rule in the 3rd dimension.
 * @tparam kC  The number of function components.
 * @tparam kL  Formulate in local (parametric) space or not.
 */
template <class Gx, class Gy, class Gz, int kC, bool kL = false>
class Hexahedron {
 public:
  static constexpr bool kLocal = kL;
  using GaussX = Gx;
  using GaussY = Gy;
  using GaussZ = Gz;
  using Gauss = gauss::Hexahedron<Gx, Gy, Gz>;
  using Scalar = typename Gauss::Scalar;
  using Local = typename Gauss::Local;
  using Global = typename Gauss::Global;
  using GaussBase = gauss::Cell<Scalar>;
  using Lagrange = typename Gauss::Lagrange;
  using Jacobian = typename Lagrange::Jacobian;
  static constexpr int Px = Gx::Q - 1;
  static constexpr int Py = Gy::Q - 1;
  static constexpr int Pz = Gz::Q - 1;
  static constexpr int P = std::max({Px, Py, Pz});
  using Basis = basis::lagrange::Hexahedron<Scalar, Px, Py, Pz>;
  static constexpr int N = Basis::N;
  static constexpr int K = kC;

 protected:
  using Mat6xN = algebra::Matrix<Scalar, 6, N>;
  using Mat6xK = algebra::Matrix<Scalar, 6, K>;
  using Mat3xK = algebra::Matrix<Scalar, 3, K>;
  using Mat3x3 = algebra::Matrix<Scalar, 3, 3>;
  using Mat3x1 = algebra::Matrix<Scalar, 3, 1>;
  using Mat1x3 = algebra::Matrix<Scalar, 1, 3>;

 public:
  using Coeff = algebra::Matrix<Scalar, K, N>;
  using Value = algebra::Matrix<Scalar, K, 1>;
  using Mat1xN = algebra::Matrix<Scalar, 1, N>;
  using Mat3xN = algebra::Matrix<Scalar, 3, N>;
  using Gradient = Mat3xK;
  using Hessian = Mat6xK;

  using GaussOnLine = GaussX;

 private:
  const Gauss *gauss_ptr_ = nullptr;
  Coeff coeff_;  // u^h(local) = coeff_ @ basis.GetValues(local)

  struct E { };

  // cache for (kLocal == true)
  /* \f$ \det(\mathbf{J}) \f$ */
  [[no_unique_address]] std::conditional_t<kLocal, std::array<Scalar, N>, E>
      jacobian_det_;
  /* \f$ \det(\mathbf{J})\,\mathbf{J}^{-1} \f$ */
  [[no_unique_address]] std::conditional_t<kLocal, std::array<Jacobian, N>, E>
      jacobian_det_inv_;
  /* \f$ \begin{bmatrix}\partial_{\xi}\\ \partial_{\eta}\\ \partial_{\zeta} \end{bmatrix}\det(\mathbf{J}) \f$ */
  [[no_unique_address]] std::conditional_t<kLocal, std::array<Local, N>, E>
      jacobian_det_grad_;
  /* \f$ \underline{J}^{-T}\,J^{-1} \f$ */
  [[no_unique_address]] std::conditional_t<kLocal, Jacobian[N], E>
      mat_after_hess_of_U_;
  /* \f$ \begin{bmatrix}\partial_{\xi}\\ \partial_{\eta}\\ \partial_{\zeta} \end{bmatrix} \qty(\underline{J}^{-T}\,J^{-1}) \f$ */
  [[no_unique_address]] std::conditional_t<kLocal, Jacobian[N][3], E>
      mat_after_grad_of_U_;
  /* \f$ \underline{C}=\begin{bmatrix}\partial_{\xi}\,J & \partial_{\eta}\,J\end{bmatrix}\underline{J}^{-T}\,J^{-2} \f$ */
  [[no_unique_address]] std::conditional_t<kLocal, Mat1x3[N], E>
      mat_before_grad_of_U_;
  [[no_unique_address]] std::conditional_t<kLocal, Jacobian[N], E>
      mat_before_U_;

  // cache for (kLocal == false)
  [[no_unique_address]] std::conditional_t<kLocal, E, std::array<Mat3xN, N>>
      basis_global_gradients_;

  static void CheckSize() {
    constexpr size_t large_member_size = kLocal
        ? sizeof(std::array<Scalar, N>) + sizeof(std::array<Jacobian, N>)
            + sizeof(std::array<Local, N>)
            + sizeof(Jacobian[N]) + sizeof(Jacobian[N][3])
            + sizeof(Jacobian[N]) + sizeof(Local[N])
        : sizeof(std::array<Mat3xN, N>);
    constexpr size_t all_member_size = large_member_size
        + sizeof(gauss_ptr_) + sizeof(coeff_);
    static_assert(sizeof(Hexahedron) >= all_member_size);
    static_assert(sizeof(Hexahedron) <= all_member_size + 16);
  }

  static const Basis basis_;
  static Basis BuildInterpolationBasis() {
    CheckSize();
    auto line_x = typename Basis::LineX{ Gauss::GaussX::BuildPoints() };
    auto line_y = typename Basis::LineY{ Gauss::GaussY::BuildPoints() };
    auto line_z = typename Basis::LineZ{ Gauss::GaussZ::BuildPoints() };
    return Basis(line_x, line_y, line_z);
  }

  static const std::array<Mat3xN, N> basis_local_gradients_;
  static std::array<Mat3xN, N> BuildBasisLocalGradients() {
    std::array<Mat3xN, N> gradients;
    auto basis = BuildInterpolationBasis();
    for (int ijk = 0; ijk < N; ++ijk) {
      auto &grad = gradients[ijk];
      auto [i, j, k] = basis.index(ijk);
      grad.row(0) = basis.GetDerivatives(1, 0, 0, i, j, k);
      grad.row(1) = basis.GetDerivatives(0, 1, 0, i, j, k);
      grad.row(2) = basis.GetDerivatives(0, 0, 1, i, j, k);
    }
    return gradients;
  }
  static const std::array<Mat6xN, N> basis_local_hessians_;
  static std::array<Mat6xN, N> BuildBasisLocalHessians() {
    std::array<Mat6xN, N> hessians;
    auto basis = BuildInterpolationBasis();
    for (int ijk = 0; ijk < N; ++ijk) {
      auto &hess = hessians[ijk];
      auto [i, j, k] = basis.index(ijk);
      hess.row(XX) = basis.GetDerivatives(2, 0, 0, i, j, k);
      hess.row(XY) = basis.GetDerivatives(1, 1, 0, i, j, k);
      hess.row(XZ) = basis.GetDerivatives(1, 0, 1, i, j, k);
      hess.row(YY) = basis.GetDerivatives(0, 2, 0, i, j, k);
      hess.row(YZ) = basis.GetDerivatives(0, 1, 1, i, j, k);
      hess.row(ZZ) = basis.GetDerivatives(0, 0, 2, i, j, k);
    }
    return hessians;
  }

 public:
  explicit Hexahedron(const GaussBase &gauss) requires(kLocal)
      : gauss_ptr_(dynamic_cast<const Gauss *>(&gauss)) {
    for (int ijk = 0; ijk < N; ++ijk) {
      auto &local = gauss_ptr_->GetLocalCoord(ijk);
      Jacobian mat = coordinate().LocalToJacobian(local);
      Jacobian inv = mat.inverse();
      Scalar det = mat.determinant();
      jacobian_det_[ijk] = det;
      jacobian_det_inv_[ijk] = det * inv;
      jacobian_det_grad_[ijk]
          = coordinate().LocalToJacobianDeterminantGradient(local);
      // cache for evaluating Hessian
      Jacobian inv_T = inv.transpose();
      mat_after_hess_of_U_[ijk] = inv_T / det;
      auto mat_grad = coordinate().LocalToJacobianGradient(local);
      Jacobian inv_T_grad[3];
      inv_T_grad[X] = -(inv * mat_grad[X] * inv).transpose();
      inv_T_grad[Y] = -(inv * mat_grad[Y] * inv).transpose();
      inv_T_grad[Z] = -(inv * mat_grad[Z] * inv).transpose();
      Mat3x1 const &det_grad = jacobian_det_grad_[ijk];
      Scalar det2 = det * det;
      mat_after_grad_of_U_[ijk][X] = inv_T_grad[X] / det
          + inv_T * (-det_grad[X] / det2);
      mat_after_grad_of_U_[ijk][Y] = inv_T_grad[Y] / det
          + inv_T * (-det_grad[Y] / det2);
      mat_after_grad_of_U_[ijk][Z] = inv_T_grad[Z] / det
          + inv_T * (-det_grad[Z] / det2);
      mat_before_grad_of_U_[ijk] = det_grad.transpose() * inv_T / det2;
      auto det_hess = coordinate().LocalToJacobianDeterminantHessian(local);
      auto &mat_before_U = mat_before_U_[ijk];
      mat_before_U(X, X) = det_hess[XX];
      mat_before_U(X, Y) = det_hess[XY];
      mat_before_U(X, Z) = det_hess[XZ];
      mat_before_U(Y, X) = det_hess[YX];
      mat_before_U(Y, Y) = det_hess[YY];
      mat_before_U(Y, Z) = det_hess[YZ];
      mat_before_U(Z, X) = det_hess[ZX];
      mat_before_U(Z, Y) = det_hess[ZY];
      mat_before_U(Z, Z) = det_hess[ZZ];
      mat_before_U *= inv_T / det2;
      Scalar det3 = det2 * det;
      mat_before_U.row(X) += det_grad.transpose() *
          (inv_T_grad[X] / det2 + inv_T * (-2 * det_grad[X] / det3));
      mat_before_U.row(Y) += det_grad.transpose() *
          (inv_T_grad[Y] / det2 + inv_T * (-2 * det_grad[Y] / det3));
      mat_before_U.row(Z) += det_grad.transpose() *
          (inv_T_grad[Z] / det2 + inv_T * (-2 * det_grad[Z] / det3));
    }
  }
  explicit Hexahedron(const GaussBase &gauss) requires(!kLocal)
      : gauss_ptr_(dynamic_cast<const Gauss *>(&gauss)) {
    for (int ijk = 0; ijk < N; ++ijk) {
      auto &local = gauss_ptr_->GetLocalCoord(ijk);
      Jacobian jacobian = coordinate().LocalToJacobian(local);
      basis_global_gradients_[ijk] = LocalGradientsToGlobalGradients(
          jacobian, basis_local_gradients_[ijk]);
    }
  }
  Hexahedron() = default;
  Hexahedron(const Hexahedron &) = default;
  Hexahedron(Hexahedron &&) noexcept = default;
  Hexahedron &operator=(const Hexahedron &) = default;
  Hexahedron &operator=(Hexahedron &&) noexcept = default;
  ~Hexahedron() noexcept = default;

  Value LocalToValue(Local const &local) const requires(kLocal) {
    Value value = coeff_ * basis_.GetValues(local).transpose();
    value /= coordinate().LocalToJacobian(local).determinant();
    return value;
  }
  Value LocalToValue(Local const &local) const requires(!kLocal) {
    return coeff_ * basis_.GetValues(local).transpose();
  }
  void LocalToGlobalAndValue(Local const &local,
      Global *global, Value *value) const {
    *global = gauss().coordinate().LocalToGlobal(local);
    *value = LocalToValue(local);
  }

  Value GlobalToValue(Global const &global) const {
    Local local = coordinate().GlobalToLocal(global);
    return LocalToValue(local);
  }
  /**
   * @brief Get the value of \f$ u(x,y,z) \equiv \det(\mathbf{J})^{-1}\,U(\xi,\eta,\zeta) \f$ at a Gaussian point.
   * 
   * This version is compiled only if `kLocal` is `true`.
   * 
   * @param ijk the index of the Gaussian point
   * @return Value the value \f$ u(x_i,y_i,z_i) \f$
   */
  Value GetValue(int i) const requires(kLocal) {
    return coeff_.col(i) / jacobian_det_[i];
  }
  void SetValue(int i, Value const &value) requires(kLocal) {
    coeff_.col(i) = value;  // value in physical space
    coeff_.col(i) *= jacobian_det_[i];  // value in parametric space
  }
  /**
   * @brief Get the value of \f$ u(x,y,z) \f$ at a Gaussian point.
   * 
   * This version is compiled only if `kLocal` is `false`.
   * 
   * @param ijk the index of the Gaussian point
   * @return Value the value \f$ u(x_i,y_i,z_i) \f$
   */
  Value GetValue(int i) const requires(!kLocal) {
    return coeff_.col(i);
  }
  void SetValue(int i, Value const &value) requires(!kLocal) {
    coeff_.col(i) = value;
  }
  Mat1xN GlobalToBasisValues(Global const &global) const {
    Local local = coordinate().GlobalToLocal(global);
    return basis_.GetValues(local);
  }
  Mat3xN GlobalToBasisGradients(Global const &global) const {
    Local local = coordinate().GlobalToLocal(global);
    Mat3xN grad;
    grad.row(0) = basis_.GetDerivatives(1, 0, 0, local);
    grad.row(1) = basis_.GetDerivatives(0, 1, 0, local);
    grad.row(2) = basis_.GetDerivatives(0, 0, 1, local);
    Jacobian jacobian = coordinate().LocalToJacobian(local);
    return LocalGradientsToGlobalGradients(jacobian, grad);
  }
  /**
   * @brief Get the local gradients of basis at a Gaussian point.
   * 
   * This version is compiled only if `kLocal` is `true`.
   * 
   * @param ijk the index of the Gaussian point
   * @return const Mat3xN& the local gradients of basis
   */
  const Mat3xN &GetBasisGradients(int ijk) const requires(kLocal) {
    return basis_local_gradients_[ijk];
  }
  /**
   * @brief Get the global gradients of basis at a Gaussian point.
   * 
   * This version is compiled only if `kLocal` is `false`.
   * 
   * @param ijk the index of the Gaussian point
   * @return const Mat3xN& the global gradients of basis
   */
  const Mat3xN &GetBasisGradients(int ijk) const requires(!kLocal) {
    return basis_global_gradients_[ijk];
  }
  /**
   * @brief Get \f$ \begin{bmatrix}\partial_{\xi}\\ \partial_{\eta}\\ \cdots \end{bmatrix} U \f$ at a Gaussian point.
   * 
   */
  Gradient GetLocalGradient(int ijk) const requires(kLocal) {
    Gradient value_grad; value_grad.setZero();
    Mat3xN const &basis_grads = GetBasisGradients(ijk);
    for (int abc = 0; abc < N; ++abc) {
      value_grad += basis_grads.col(abc) * coeff_.col(abc).transpose();
    }
    return value_grad;
  }
  Gradient LocalToLocalGradient(Local const &local) const requires(kLocal) {
    Gradient value_grad; value_grad.setZero();
    auto x = local[X], y = local[Y], z = local[Z];
    Mat3xN basis_grad;
    basis_grad.row(X) = basis_.GetDerivatives(1, 0, 0, x, y, z);
    basis_grad.row(Y) = basis_.GetDerivatives(0, 1, 0, x, y, z);
    basis_grad.row(Z) = basis_.GetDerivatives(0, 0, 1, x, y, z);
    for (int abc = 0; abc < N; ++abc) {
      value_grad += basis_grad.col(abc) * coeff_.col(abc).transpose();
    }
    return value_grad;
  }
  Gradient LocalToGlobalGradient(Local const &local) const requires(kLocal) {
    Gradient grad = LocalToLocalGradient(local);
    Jacobian mat = coordinate().LocalToJacobian(local);
    Scalar det = mat.determinant();
    Global det_grad = coordinate().LocalToJacobianDeterminantGradient(local);
    Value value = coeff_ * basis_.GetValues(local).transpose();
    grad -= (det_grad / det) * value.transpose();
    return mat.inverse() / det * grad;
  }
  Gradient GlobalToGlobalGradient(Global const &global) const requires(kLocal) {
    auto local = coordinate().GlobalToLocal(global);
    return LocalToGlobalGradient(local);
  }
  /**
   * @brief Get \f$ \begin{bmatrix}\partial_{x}\\ \partial_{y}\\ \cdots \end{bmatrix} u \f$ at a Gaussian point.
   * 
   * This version is compiled only if `kLocal` is `false`.
   * 
   */
  Gradient GetGlobalGradient(int ijk) const requires(!kLocal) {
    Mat3xN const &basis_grad = GetBasisGradients(ijk);
    return basis_grad * coeff().transpose();
  }
  /**
   * @brief Get \f$ \begin{bmatrix}\partial_{x}\\ \partial_{y}\\ \cdots \end{bmatrix} u \f$ at a Gaussian point.
   * 
   * This version is compiled only if `kLocal` is `true`.
   * 
   */
  Gradient GetGlobalGradient(int ijk) const requires(kLocal) {
    return GetGlobalGradient(GetValue(ijk), GetLocalGradient(ijk), ijk);
  }
  Gradient GetGlobalGradient(Value const &value_ijk, Gradient local_grad_ijk,
      int ijk) const requires(kLocal) {
    auto &value_grad = local_grad_ijk;
    value_grad -= jacobian_det_grad_[ijk] * value_ijk.transpose();
    auto jacobian_det = jacobian_det_[ijk];
    value_grad /= (jacobian_det * jacobian_det);
    return GetJacobianAssociated(ijk) * value_grad;
  }
  /**
   * @brief Get the local Hessians of basis at a Gaussian point.
   * 
   * This version is compiled only if `kLocal` is `true`.
   * 
   * @param ijk the index of the Gaussian point
   * @return const Mat6xN& the local Hessians of basis
   */
  const Mat6xN &GetBasisHessians(int ijk) const requires(kLocal) {
    return basis_local_hessians_[ijk];
  }
  /**
   * @brief Get \f$ \begin{bmatrix}\partial_{\xi}\partial_{\xi}\\ \partial_{\xi}\partial_{\eta}\\ \cdots \end{bmatrix} U \f$ at a Gaussian point.
   * 
   */
  Hessian GetLocalHessian(int ijk) const requires(kLocal) {
    Mat6xK value_hess; value_hess.setZero();
    Mat6xN const &basis_hess = GetBasisHessians(ijk);
    for (int abc = 0; abc < N; ++abc) {
      value_hess += basis_hess.col(abc) * coeff_.col(abc).transpose();
    }
    return value_hess;
  }
  /**
   * @brief Get \f$ \begin{bmatrix}\partial_{x}\partial_{x}\\ \partial_{x}\partial_{y}\\ \cdots \end{bmatrix} u \f$ at a Gaussian point.
   * 
   */
  Hessian GetGlobalHessian(int ijk) const requires(kLocal) {
    return GetGlobalHessian(GetLocalGradient(ijk), ijk);
  }
  Hessian GetGlobalHessian(Gradient const &local_grad_ijk, int ijk) const
      requires(kLocal) {
    Hessian local_hess = GetLocalHessian(ijk);
    auto &global_hess = local_hess;
    for (int k = 0; k < K; ++k) {
      Mat3x3 scalar_hess;
      scalar_hess(X, X) = local_hess(XX, k);
      scalar_hess(X, Y) =
      scalar_hess(Y, X) = local_hess(XY, k);
      scalar_hess(X, Z) =
      scalar_hess(Z, X) = local_hess(XZ, k);
      scalar_hess(Y, Y) = local_hess(YY, k);
      scalar_hess(Y, Z) =
      scalar_hess(Z, Y) = local_hess(YZ, k);
      scalar_hess(Z, Z) = local_hess(ZZ, k);
      scalar_hess *= mat_after_hess_of_U_[ijk];
      Mat1x3 scalar_local_grad = local_grad_ijk.col(k);
      scalar_hess.row(X) += scalar_local_grad * mat_after_grad_of_U_[ijk][X]
          - mat_before_grad_of_U_[ijk] * scalar_local_grad[X];
      scalar_hess.row(Y) += scalar_local_grad * mat_after_grad_of_U_[ijk][Y]
          - mat_before_grad_of_U_[ijk] * scalar_local_grad[Y];
      scalar_hess.row(Z) += scalar_local_grad * mat_after_grad_of_U_[ijk][Z]
          - mat_before_grad_of_U_[ijk] * scalar_local_grad[Z];
      Scalar scalar_local_val = coeff_(k, ijk);
      scalar_hess -= mat_before_U_[ijk] * scalar_local_val;
      scalar_hess = jacobian_det_inv_[ijk] * scalar_hess;
      scalar_hess /= jacobian_det_[ijk];
      global_hess(XX, k) = scalar_hess(X, X);
      global_hess(XY, k) = scalar_hess(X, Y);
      global_hess(XZ, k) = scalar_hess(X, Z);
      global_hess(YY, k) = scalar_hess(Y, Y);
      global_hess(YZ, k) = scalar_hess(Y, Z);
      global_hess(ZZ, k) = scalar_hess(Z, Z);
    }
    return global_hess;
  }

  /**
   * @brief Convert the gradients in local coordinates to the gradients in global coordinates.
   * 
   * \f$ \begin{bmatrix}\partial_{\xi}\,\phi\\\partial_{\eta}\,\phi\\\partial_{\zeta}\,\phi\end{bmatrix}=\mathbf{J}\begin{bmatrix}\partial_{x}\,\phi\\\partial_{y}\,\phi\\\partial_{z}\,\phi\end{bmatrix} \f$, in which \f$ \mathbf{J}=\begin{bmatrix}\partial_{\xi}\\\partial_{\eta}\\\partial_{\zeta}\end{bmatrix}\begin{bmatrix}x & y & z\end{bmatrix} \f$ is `geometry::Element::Jacobian`.
   * 
   * @param jacobian the Jacobian matrix of the type `geometry::Element::Jacobian`.
   * @param local_grad the gradients in local coordinates
   * @return Mat3xN the gradients in global coordinates
   */
  static Mat3xN LocalGradientsToGlobalGradients(const Jacobian &jacobian,
      Mat3xN const &local_grad) {
    return jacobian.fullPivLu().solve(local_grad);
  }

  /**
   * @brief Convert a flux matrix from global to local at a given Gaussian point.
   * 
   * \f$ \begin{bmatrix}F^{\xi} & F^{\eta} & F^{\zeta}\end{bmatrix}=\begin{bmatrix}f^{x} & f^{y} & f^{z}\end{bmatrix}\mathbf{J}^{*} \f$, in which \f$ \mathbf{J}^{*} = \det(\mathbf{J}) \begin{bmatrix}\partial_{x}\\\partial_{y}\\\partial_{z}\end{bmatrix}\begin{bmatrix}\xi & \eta & \zeta\end{bmatrix} \f$ is returned by `Hexahedron::GetJacobianAssociated`.
   * 
   * @tparam FluxMatrix a matrix type which has 3 columns
   * @param global_flux the global flux
   * @param ijk the index of the Gaussian point
   * @return FluxMatrix the local flux
   */
  template <class FluxMatrix>
  FluxMatrix GlobalFluxToLocalFlux(const FluxMatrix &global_flux, int ijk) const
      requires(kLocal) {
    FluxMatrix local_flux = global_flux * GetJacobianAssociated(ijk);
    return local_flux;
  }

  /**
   * @brief Get the associated matrix of the Jacobian at a given Gaussian point.
   * 
   * \f$ \mathbf{J}^{*}=\det(\mathbf{J})\,\mathbf{J}^{-1} \f$, in which \f$ \mathbf{J}^{-1}=\begin{bmatrix}\partial_{x}\\\partial_{y}\\\partial_{z}\end{bmatrix}\begin{bmatrix}\xi & \eta & \zeta\end{bmatrix} \f$ is the inverse of `geometry::Element::Jacobian`.
   * 
   * @param ijk the index of the Gaussian point
   * @return Jacobian const& the associated matrix of \f$ \mathbf{J} \f$.
   */
  Jacobian const &GetJacobianAssociated(int ijk) const
      requires(kLocal) {
    return jacobian_det_inv_[ijk];
  }

  Global const &center() const {
    return gauss_ptr_->center();
  }
  Coeff const &coeff() const {
    return coeff_;
  }
  Coeff &coeff() {
    return coeff_;
  }
  Basis const &basis() const {
    return basis_;
  }
  Gauss const &gauss() const {
    return *gauss_ptr_;
  }
  Lagrange const &coordinate() const {
    return gauss().coordinate();
  }
  template <typename Callable>
  void Approximate(Callable &&global_to_value) {
    for (int ijk = 0; ijk < N; ++ijk) {
      const auto &global = gauss_ptr_->GetGlobalCoord(ijk);
      SetValue(ijk, global_to_value(global));
    }
  }
  const Scalar *GetCoeffFrom(const Scalar *input) {
    std::memcpy(coeff_.data(), input, sizeof(Scalar) * coeff_.size());
    return input + coeff_.size();
  }
  Scalar *WriteCoeffTo(Scalar *output) const {
    std::memcpy(output, coeff_.data(), sizeof(Scalar) * coeff_.size());
    return output + coeff_.size();
  }
  /**
   * @brief Add the given Coeff to the dofs corresponding to the given basis.
   * 
   * @param coeff the coeff to be added
   * @param output the beginning of all dofs
   */
  static void AddCoeffTo(Coeff const &coeff, Scalar *output) {
    for (int c = 0; c < N; ++c) {
      for (int r = 0; r < K; ++r) {
        *output++ += coeff(r, c);
      }
    }
  }
  static void MinusCoeff(Coeff const &coeff, Scalar *output) {
    for (int c = 0; c < N; ++c) {
      for (int r = 0; r < K; ++r) {
        *output++ -= coeff(r, c);
      }
    }
  }
  /**
   * @brief Add the given Value to the dofs corresponding to the given basis.
   * 
   * @param value the value to be added
   * @param output the beginning of all dofs
   * @param i_basis the (0-based) index of basis
   * @return Scalar* the end of the modified column 
   */
  static Scalar *AddValueTo(Value const &value, Scalar *output, int i_basis) {
    assert(0 <= i_basis && i_basis < N);
    output += K * i_basis;
    for (int r = 0; r < K; ++r) {
      *output++ += value[r];
    }
    return output;
  }
  static Scalar *MinusValue(Value const &value, Scalar *output, int i_basis) {
    assert(0 <= i_basis && i_basis < N);
    output += K * i_basis;
    for (int r = 0; r < K; ++r) {
      *output++ -= value[r];
    }
    return output;
  }
  /**
   * @brief Multiply the given scale to the Value at the given address.
   * 
   * @param scale the scale to be multiplied
   * @param output the address of the value
   * @return the address of the next value
   */
  static Scalar *ScaleValueAt(double scale, Scalar *output) {
    for (int r = 0; r < K; ++r) {
      *output++ *= scale;
    }
    return output;
  }

  int FindFaceId(Global const &face_center) const {
    int i_face;
    auto almost_equal = [&face_center](Global point) {
      point -= face_center;
      return point.norm() < 1e-10;
    };
    if (almost_equal(coordinate().LocalToGlobal(0, 0, -1))) { i_face = 0; }
    else if (almost_equal(coordinate().LocalToGlobal(0, -1, 0))) { i_face = 1; }
    else if (almost_equal(coordinate().LocalToGlobal(+1, 0, 0))) { i_face = 2; }
    else if (almost_equal(coordinate().LocalToGlobal(0, +1, 0))) { i_face = 3; }
    else if (almost_equal(coordinate().LocalToGlobal(-1, 0, 0))) { i_face = 4; }
    else if (almost_equal(coordinate().LocalToGlobal(0, 0, +1))) { i_face = 5; }
    else { assert(false); }
    return i_face;
  }
  std::vector<int> FindCollinearPoints(Global const &global, int i_face) const {
    std::vector<int> indices;
    using mini::geometry::X;
    using mini::geometry::Y;
    using mini::geometry::Z;
    auto local = coordinate().GlobalToLocal(global);
    int i, j, k;
    auto almost_equal = [](Scalar x, Scalar y) {
      return std::abs(x - y) < 1e-10;
    };
    switch (i_face) {
    case 0:
      assert(almost_equal(local[Z], -1));
      for (i = 0; i < GaussX::Q; ++i) {
        if (almost_equal(local[X], GaussX::points[i])) {
          break;
        }
      }
      for (j = 0; j < GaussY::Q; ++j) {
        if (almost_equal(local[Y], GaussY::points[j])) {
          break;
        }
      }
      for (k = 0; k < GaussZ::Q; ++k) {
        indices.push_back(basis().index(i, j, k));
      }
      break;
    case 1:
      assert(almost_equal(local[Y], -1));
      for (i = 0; i < GaussX::Q; ++i) {
        if (almost_equal(local[X], GaussX::points[i])) {
          break;
        }
      }
      for (k = 0; k < GaussZ::Q; ++k) {
        if (almost_equal(local[Z], GaussZ::points[k])) {
          break;
        }
      }
      for (j = 0; j < GaussY::Q; ++j) {
        indices.push_back(basis().index(i, j, k));
      }
      break;
    case 2:
      assert(almost_equal(local[X], +1));
      for (j = 0; j < GaussY::Q; ++j) {
        if (almost_equal(local[Y], GaussY::points[j])) {
          break;
        }
      }
      for (k = 0; k < GaussZ::Q; ++k) {
        if (almost_equal(local[Z], GaussZ::points[k])) {
          break;
        }
      }
      for (i = 0; i < GaussX::Q; ++i) {
        indices.push_back(basis().index(i, j, k));
      }
      break;
    case 3:
      assert(almost_equal(local[Y], +1));
      for (i = 0; i < GaussX::Q; ++i) {
        if (almost_equal(local[X], GaussX::points[i])) {
          break;
        }
      }
      for (k = 0; k < GaussZ::Q; ++k) {
        if (almost_equal(local[Z], GaussZ::points[k])) {
          break;
        }
      }
      for (j = 0; j < GaussY::Q; ++j) {
        indices.push_back(basis().index(i, j, k));
      }
      break;
    case 4:
      assert(almost_equal(local[X], -1));
      for (j = 0; j < GaussY::Q; ++j) {
        if (almost_equal(local[Y], GaussY::points[j])) {
          break;
        }
      }
      for (k = 0; k < GaussZ::Q; ++k) {
        if (almost_equal(local[Z], GaussZ::points[k])) {
          break;
        }
      }
      for (i = 0; i < GaussX::Q; ++i) {
        indices.push_back(basis().index(i, j, k));
      }
      break;
    case 5:
      assert(almost_equal(local[Z], +1));
      for (i = 0; i < GaussX::Q; ++i) {
        if (almost_equal(local[X], GaussX::points[i])) {
          break;
        }
      }
      for (j = 0; j < GaussY::Q; ++j) {
        if (almost_equal(local[Y], GaussY::points[j])) {
          break;
        }
      }
      for (k = 0; k < GaussZ::Q; ++k) {
        indices.push_back(basis().index(i, j, k));
      }
      break;
    default: assert(false);
    }
    return indices;
  }
  std::tuple<int, int, int> FindCollinearIndex(Global const &global, int i_face) const {
    return FindCollinearIndexByGlobal(global, i_face);
  }
  std::tuple<int, int, int> FindCollinearIndexByGlobal(Global const &global, int i_face) const {
    int i{-1}, j{-1}, k{-1};
    using mini::geometry::X;
    using mini::geometry::Y;
    using mini::geometry::Z;
    Global global_temp;
    bool done = false;
    switch (i_face) {
    case 0:
      for (i = 0; i < GaussX::Q; ++i) {
        for (j = 0; j < GaussY::Q; ++j) {
          global_temp = coordinate().LocalToGlobal(
              GaussX::points[i], GaussY::points[j], -1);
          global_temp -= global;
          if (global_temp.norm() < 1e-8) {
            done = true;
            break;
          }
        }
        if (done) {
          break;
        }
      }
      break;
    case 1:
      for (i = 0; i < GaussX::Q; ++i) {
        for (k = 0; k < GaussZ::Q; ++k) {
          global_temp = coordinate().LocalToGlobal(
              GaussX::points[i], -1, GaussZ::points[k]);
          global_temp -= global;
          if (global_temp.norm() < 1e-8) {
            done = true;
            break;
          }
        }
        if (done) {
          break;
        }
      }
      break;
    case 2:
      for (j = 0; j < GaussY::Q; ++j) {
        for (k = 0; k < GaussZ::Q; ++k) {
          global_temp = coordinate().LocalToGlobal(
              +1, GaussY::points[j], GaussZ::points[k]);
          global_temp -= global;
          if (global_temp.norm() < 1e-8) {
            done = true;
            break;
          }
        }
        if (done) {
          break;
        }
      }
      break;
    case 3:
      for (i = 0; i < GaussX::Q; ++i) {
        for (k = 0; k < GaussZ::Q; ++k) {
          global_temp = coordinate().LocalToGlobal(
              GaussX::points[i], +1, GaussZ::points[k]);
          global_temp -= global;
          if (global_temp.norm() < 1e-8) {
            done = true;
            break;
          }
        }
        if (done) {
          break;
        }
      }
      break;
    case 4:
      for (j = 0; j < GaussY::Q; ++j) {
        for (k = 0; k < GaussZ::Q; ++k) {
          global_temp = coordinate().LocalToGlobal(
              -1, GaussY::points[j], GaussZ::points[k]);
          global_temp -= global;
          if (global_temp.norm() < 1e-8) {
            done = true;
            break;
          }
        }
        if (done) {
          break;
        }
      }
      break;
    case 5:
      for (i = 0; i < GaussX::Q; ++i) {
        for (j = 0; j < GaussY::Q; ++j) {
          global_temp = coordinate().LocalToGlobal(
              GaussX::points[i], GaussY::points[j], +1);
          global_temp -= global;
          if (global_temp.norm() < 1e-8) {
            done = true;
            break;
          }
        }
        if (done) {
          break;
        }
      }
      break;
    default: assert(false);
    }
    return std::make_tuple(i, j, k);
  }
  std::tuple<int, int, int> FindCollinearIndexByLocal(Global const &global, int i_face) const {
    int i{-1}, j{-1}, k{-1};
    using mini::geometry::X;
    using mini::geometry::Y;
    using mini::geometry::Z;
    Local local_hint(0, 0, 0);
    switch (i_face) {
    case 0: local_hint[Z] = -1; break;
    case 1: local_hint[Y] = -1; break;
    case 2: local_hint[X] = +1; break;
    case 3: local_hint[Y] = +1; break;
    case 4: local_hint[X] = -1; break;
    case 5: local_hint[Z] = +1; break;
    default: assert(false);
    }
    auto local = coordinate().GlobalToLocal(global, local_hint);
    auto almost_equal = [](Scalar x, Scalar y) {
      return std::abs(x - y) < 1e-10;
    };
    switch (i_face) {
    case 0:
      assert(almost_equal(local[Z], -1));
      for (i = 0; i < GaussX::Q; ++i) {
        if (almost_equal(local[X], GaussX::points[i])) {
          break;
        }
      }
      for (j = 0; j < GaussY::Q; ++j) {
        if (almost_equal(local[Y], GaussY::points[j])) {
          break;
        }
      }
      break;
    case 1:
      assert(almost_equal(local[Y], -1));
      for (i = 0; i < GaussX::Q; ++i) {
        if (almost_equal(local[X], GaussX::points[i])) {
          break;
        }
      }
      for (k = 0; k < GaussZ::Q; ++k) {
        if (almost_equal(local[Z], GaussZ::points[k])) {
          break;
        }
      }
      break;
    case 2:
      assert(almost_equal(local[X], +1));
      for (j = 0; j < GaussY::Q; ++j) {
        if (almost_equal(local[Y], GaussY::points[j])) {
          break;
        }
      }
      for (k = 0; k < GaussZ::Q; ++k) {
        if (almost_equal(local[Z], GaussZ::points[k])) {
          break;
        }
      }
      break;
    case 3:
      assert(almost_equal(local[Y], +1));
      for (i = 0; i < GaussX::Q; ++i) {
        if (almost_equal(local[X], GaussX::points[i])) {
          break;
        }
      }
      for (k = 0; k < GaussZ::Q; ++k) {
        if (almost_equal(local[Z], GaussZ::points[k])) {
          break;
        }
      }
      break;
    case 4:
      assert(almost_equal(local[X], -1));
      for (j = 0; j < GaussY::Q; ++j) {
        if (almost_equal(local[Y], GaussY::points[j])) {
          break;
        }
      }
      for (k = 0; k < GaussZ::Q; ++k) {
        if (almost_equal(local[Z], GaussZ::points[k])) {
          break;
        }
      }
      break;
    case 5:
      assert(almost_equal(local[Z], +1));
      for (i = 0; i < GaussX::Q; ++i) {
        if (almost_equal(local[X], GaussX::points[i])) {
          break;
        }
      }
      for (j = 0; j < GaussY::Q; ++j) {
        if (almost_equal(local[Y], GaussY::points[j])) {
          break;
        }
      }
      break;
    default: assert(false);
    }
    return std::make_tuple(i, j, k);
  }
};
template <class Gx, class Gy, class Gz, int kC, bool kL>
typename Hexahedron<Gx, Gy, Gz, kC, kL>::Basis const
Hexahedron<Gx, Gy, Gz, kC, kL>::basis_ =
    Hexahedron<Gx, Gy, Gz, kC, kL>::BuildInterpolationBasis();

template <class Gx, class Gy, class Gz, int kC, bool kL>
std::array<typename Hexahedron<Gx, Gy, Gz, kC, kL>::Mat3xN,
                    Hexahedron<Gx, Gy, Gz, kC, kL>::N> const
Hexahedron<Gx, Gy, Gz, kC, kL>::basis_local_gradients_ =
    Hexahedron<Gx, Gy, Gz, kC, kL>::BuildBasisLocalGradients();

template <class Gx, class Gy, class Gz, int kC, bool kL>
std::array<typename Hexahedron<Gx, Gy, Gz, kC, kL>::Mat6xN,
                    Hexahedron<Gx, Gy, Gz, kC, kL>::N> const
Hexahedron<Gx, Gy, Gz, kC, kL>::basis_local_hessians_ =
    Hexahedron<Gx, Gy, Gz, kC, kL>::BuildBasisLocalHessians();

}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_HEXAHEDRON_HPP_
