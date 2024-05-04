// Copyright 2021 PEI Weicheng and JIANG Yuyan
/**
 * This file defines parser of partition info txt.
 */
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

namespace std {

template <class Matrix>
Matrix abs(Matrix const &matrix)
    requires(!std::is_scalar_v<Matrix>) {
  return matrix.array().abs();
}

template <class Matrix>
Matrix pow(Matrix const &matrix, int e)
    requires(!std::is_scalar_v<Matrix>) {
  using Scalar = decltype(*matrix.data());
  return matrix.unaryExpr([e](Scalar x){ return std::pow(x, e); });
}

}  // namespace std

#endif  // MINI_ALGEBRA_EIGEN_HPP_
