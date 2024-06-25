// Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_ALGEBRA_EIGEN_HPP_
#define MINI_ALGEBRA_EIGEN_HPP_

#include "Eigen/Eigen"

namespace mini {
namespace algebra {

using Eigen::Array;
using Eigen::Matrix;
using Eigen::Vector;

template <typename Scalar>
using DynamicMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Scalar>
using DynamicVector = Eigen::Vector<Scalar, Eigen::Dynamic>;

/**
 * @brief Set the value of a scalar to be 0.
 * 
 * @tparam Scalar the type of the scalar
 * @param s the address of the scalar
 */
template <std::floating_point Scalar>
inline void SetZero(Scalar *s) {
  *s = 0;
}

/**
 * @brief Set all coefficients of a matrix to be 0.
 * 
 * @tparam Scalar the type of the matrix's coefficient
 * @tparam M the number of rows of the matrix
 * @tparam N the number of columns of the matrix
 * @param m the address of the matrix
 */
template <std::floating_point Scalar, int M, int N>
inline void SetZero(algebra::Matrix<Scalar, M, N>* m) {
  m->setZero();
}

template <class Matrix>
void Maximize(Matrix *lhs, Matrix const &rhs)
    requires(!std::is_scalar_v<Matrix>) {
  auto *l_data = lhs->data();
  auto const *r_data = rhs.data();
  for (int k = 0, K = lhs->size(); k < K; ++k) {
    l_data[k] = std::max(l_data[k], r_data[k]);
  }
}

template <class Matrix>
void Minimize(Matrix *lhs, Matrix const &rhs)
    requires(!std::is_scalar_v<Matrix>) {
  auto *l_data = lhs->data();
  auto const *r_data = rhs.data();
  for (int k = 0, K = lhs->size(); k < K; ++k) {
    l_data[k] = std::min(l_data[k], r_data[k]);
  }
}

template <class MatrixType>
using LowerTriangularView = Eigen::TriangularView<MatrixType, Eigen::Lower>;

template <class MatrixType>
auto GetLowerTriangularView(MatrixType const &matrix) {
  return matrix.template triangularView<Eigen::Lower>();
}
template <class MatrixType>
auto GetLowerTriangularView(MatrixType *matrix) {
  return matrix->template triangularView<Eigen::Lower>();
}

}  // namespace algebra
}  // namespace mini

/**
 * @brief Extends utilities defined in the C++ standard library.
 * 
 */
namespace std {

template <class Matrix>
Matrix abs(Matrix const &matrix)
    requires(!std::is_scalar_v<Matrix>) {
  return matrix.array().abs();
}

template <class Matrix, class E>
Matrix pow(Matrix const &matrix, E e)
    requires(!std::is_scalar_v<Matrix> && std::is_scalar_v<E>) {
  using Scalar = decltype(*matrix.data());
  return matrix.unaryExpr([e](Scalar x){ return std::pow(x, e); });
}

}  // namespace std

#endif  // MINI_ALGEBRA_EIGEN_HPP_
