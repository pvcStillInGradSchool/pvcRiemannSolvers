//  Copyright 2023 PEI Weicheng
#ifndef MINI_BASIS_TAYLOR_HPP_
#define MINI_BASIS_TAYLOR_HPP_

#include <concepts>

#include <cassert>
#include <cmath>

#include <iostream>
#include <type_traits>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace basis {

/**
 * @brief The basis, formed by monomials, of the space spanned by polynomials.
 * 
 * @tparam Scalar the type of coordinates
 * @tparam kDimensions the dimension of underlying space
 * @tparam kDegrees the maximum degree of its members
 */
template <std::floating_point Scalar, int kDimensions, int kDegrees>
class Taylor;

template <std::floating_point Scalar, int kDegrees>
class Taylor<Scalar, 1, kDegrees> {
 public:
  // the maximum degree of members in this basis
  static constexpr int P = kDegrees;

  // the number of terms in this basis
  static constexpr int N = P + 1;

  using Vector = algebra::Vector<Scalar, N>;

  /**
   * @brief Get the values of all basis functions at an arbitrary point.
   * 
   * @param x the coordinate of the query point
   * @return Vector the values
   */
  static Vector GetValues(Scalar x) {
    Vector vec;
    Scalar x_power = 1;
    vec[0] = x_power;
    for (int k = 1; k < N; ++k) {
      vec[k] = (x_power *= x);
    }
    assert(std::abs(x_power - std::pow(x, P)) < 1e-14);
    return vec;
  }

  /**
   * @brief Get the k-th order derivatives of all basis functions at an arbitrary point.
   * 
   * @param x the coordinate of the query point
   * @param k the order of the derivatives to be taken
   * @return Vector the derivatives
   */
  static Vector GetDerivatives(int k, Scalar x) {
    assert(0 <= k && k <= P);
    Vector vec;
    vec.setZero();  // For all j < k, there is vec[j] = 0.
    auto factorial_j = std::tgamma(Scalar(k + 1));  // factorial(j == k)
    auto factorial_j_minus_k = Scalar(1);  // factorial(j - k == 0)
    // j * (j - 1) * ... * (j - k + 1) =
    vec[k] = factorial_j / factorial_j_minus_k;
    auto x_power = Scalar(1);
    for (int j = k + 1; j < N; ++j) {
      auto j_minus_k = j - k;
      factorial_j_minus_k *= j_minus_k;
      factorial_j *= j;
      x_power *= x;
      vec[j] = x_power * factorial_j / factorial_j_minus_k;
    }
    assert(std::abs(x_power - std::pow(x, P - k)) < 1e-14);
    assert(factorial_j == std::tgamma(P + 1));
    assert(factorial_j_minus_k == std::tgamma(P - k + 1));
    return vec;
  }
};

template <std::floating_point Scalar, int kDegrees>
class Taylor<Scalar, 2, kDegrees> {
  static constexpr int GetN() {
    return ((kDegrees + 2) * (kDegrees + 1)) / 2;
  }

 public:
  static constexpr int N = GetN();  // the number of components
  using MatNx1 = algebra::Matrix<Scalar, N, 1>;
  using Coord = algebra::Matrix<Scalar, 2, 1>;

  static MatNx1 GetValue(const Coord &xy) requires(kDegrees == 1) {
    MatNx1 col = { 1, xy[0], xy[1] };
    return col;
  }

  static MatNx1 GetValue(const Coord &xy) requires(kDegrees == 2) {
    auto x = xy[0], y = xy[1];
    MatNx1 col = { 1, x, y, x * x, x * y, y * y };
    return col;
  }

  static MatNx1 GetValue(const Coord &xy)  requires(kDegrees == 3) {
    auto x = xy[0], y = xy[1];
    auto x_x = x * x, x_y = x * y, y_y = y * y;
    MatNx1 col = { 1, x, y, x_x, x_y, y_y,
        x_x * x, x_x * y, x * y_y, y * y_y };
    return col;
  }
};

template <std::floating_point Scalar>
class Taylor<Scalar, 3, 0> {
 public:
  static constexpr int N = 1;  // the number of components
  using MatNx1 = algebra::Matrix<Scalar, N, 1>;
  using Coord = algebra::Matrix<Scalar, 3, 1>;

  static MatNx1 GetValue(const Coord &xyz) {
    MatNx1 col; col(0, 0) = 1;
    return col;
  }

  template <typename MatKxN>
  static MatKxN GetPdvValue(const Coord &xyz, const MatKxN &coeff) {
    MatKxN res; res.setZero();
    return res;
  }

  template <int K>
  static auto GetGradValue(const Coord &xyz,
      const algebra::Matrix<Scalar, K, N> &coeff) {
    algebra::Matrix<Scalar, K, 3> res; res.setZero();
    return res;
  }

  template <int K>
  static auto GetSmoothness(
      const algebra::Matrix<Scalar, K, N> &integral, Scalar volume) {
    using MatKx1 = algebra::Matrix<Scalar, K, 1>;
    MatKx1 smoothness; smoothness.setZero();
    return smoothness;
  }
};

template <std::floating_point Scalar>
class Taylor<Scalar, 3, 1> {
 public:
  static constexpr int N = 4;  // the number of components
  using MatNx1 = algebra::Matrix<Scalar, N, 1>;
  using MatNx3 = algebra::Matrix<Scalar, N, 3>;
  using Coord = algebra::Matrix<Scalar, 3, 1>;

  static MatNx1 GetValue(const Coord &xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    MatNx1 col = { 1, x, y, z };
    return col;
  }
  template <typename MatKxN>
  static MatKxN GetPdvValue(const Coord &xyz, const MatKxN &coeff) {
    MatKxN res = coeff; res.col(0).setZero();
    return res;
  }

  template <int K>
  static auto GetGradValue(const Coord &xyz,
      const algebra::Matrix<Scalar, K, N> &coeff) {
    algebra::Matrix<Scalar, K, 3> res;
    // pdv_x
    res.col(0) = coeff.col(1);
    // pdv_y
    res.col(1) = coeff.col(2);
    // pdv_z
    res.col(2) = coeff.col(3);
    return res;
  }

  template <int K>
  static auto GetSmoothness(
      const algebra::Matrix<Scalar, K, N> &integral, Scalar volume) {
    using MatKx1 = algebra::Matrix<Scalar, K, 1>;
    MatKx1 smoothness = integral.col(1);
    smoothness += integral.col(2);
    smoothness += integral.col(3);
    return smoothness;
  }
};

template <std::floating_point Scalar>
class Taylor<Scalar, 3, 2> {
 public:
  static constexpr int N = 10;  // the number of components
  using MatNx1 = algebra::Matrix<Scalar, N, 1>;
  using MatNx3 = algebra::Matrix<Scalar, N, 3>;
  using Coord = algebra::Matrix<Scalar, 3, 1>;

  static MatNx1 GetValue(const Coord &xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    MatNx1 col = { 1, x, y, z, x * x, x * y, x * z, y * y, y * z, z * z };
    return col;
  }
  template <typename MatKxN>
  static MatKxN GetPdvValue(const Coord &xyz, const MatKxN &coeff) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    MatKxN res = coeff; res.col(0).setZero();
    // pdv_x
    assert(res.col(1) == coeff.col(1));
    res.col(1) += coeff.col(4) * (2 * x);
    res.col(1) += coeff.col(5) * y;
    res.col(1) += coeff.col(6) * z;
    // pdv_y
    assert(res.col(2) == coeff.col(2));
    res.col(2) += coeff.col(5) * x;
    res.col(2) += coeff.col(7) * (2 * y);
    res.col(2) += coeff.col(8) * z;
    // pdv_z
    assert(res.col(3) == coeff.col(3));
    res.col(3) += coeff.col(6) * x;
    res.col(3) += coeff.col(8) * y;
    res.col(3) += coeff.col(9) * (2 * z);
    // pdv_xx
    res.col(4) += coeff.col(4);
    assert(res.col(4) == coeff.col(4) * 2);
    // pdv_xy
    assert(res.col(5) == coeff.col(5));
    // pdv_xz
    assert(res.col(6) == coeff.col(6));
    // pdv_yy
    res.col(7) += coeff.col(7);
    assert(res.col(7) == coeff.col(7) * 2);
    // pdv_yz
    assert(res.col(8) == coeff.col(8));
    // pdv_zz
    res.col(9) += coeff.col(9);
    assert(res.col(9) == coeff.col(9) * 2);
    return res;
  }

  template <int K>
  static auto GetGradValue(const Coord &xyz,
      const algebra::Matrix<Scalar, K, N> &coeff) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    algebra::Matrix<Scalar, K, 3> res;
    // pdv_x
    res.col(0) = coeff.col(1);
    res.col(0) += coeff.col(4) * (2 * x);
    res.col(0) += coeff.col(5) * y;
    res.col(0) += coeff.col(6) * z;
    // pdv_y
    res.col(1) = coeff.col(2);
    res.col(1) += coeff.col(5) * x;
    res.col(1) += coeff.col(7) * (2 * y);
    res.col(1) += coeff.col(8) * z;
    // pdv_z
    res.col(2) = coeff.col(3);
    res.col(2) += coeff.col(6) * x;
    res.col(2) += coeff.col(8) * y;
    res.col(2) += coeff.col(9) * (2 * z);
    return res;
  }

  template <int K>
  static auto GetSmoothness(
      const algebra::Matrix<Scalar, K, N> &integral, Scalar volume) {
    using MatKx1 = algebra::Matrix<Scalar, K, 1>;
    auto w1  // weight of 1st-order partial derivatives
        = std::pow(volume, 2./3-1);
    MatKx1 smoothness = integral.col(1);
    smoothness += integral.col(2);
    smoothness += integral.col(3);
    smoothness *= w1;
    auto w2  // weight of 2nd-order partial derivatives
        = std::pow(volume, 4./3-1);
    smoothness += integral.col(4) * w2;
    smoothness += integral.col(5) * w2;
    smoothness += integral.col(6) * w2;
    smoothness += integral.col(7) * w2;
    smoothness += integral.col(8) * w2;
    smoothness += integral.col(9) * w2;
    return smoothness;
  }
};

template <std::floating_point Scalar>
class Taylor<Scalar, 3, 3> {
  static constexpr int X{1}, Y{2}, Z{3};
  static constexpr int XX{4}, XY{5}, XZ{6}, YY{7}, YZ{8}, ZZ{9};
  static constexpr int XXX{10}, XXY{11}, XXZ{12}, XYY{13}, XYZ{14}, XZZ{15};
  static constexpr int YYY{16}, YYZ{17}, YZZ{18}, ZZZ{19};

 public:
  static constexpr int N = 20;  // the number of components
  using MatNx1 = algebra::Matrix<Scalar, N, 1>;
  using MatNx3 = algebra::Matrix<Scalar, N, 3>;
  using Coord = algebra::Matrix<Scalar, 3, 1>;

  static MatNx1 GetValue(const Coord &xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    auto xx{x * x}, xy{x * y}, xz{x * z}, yy{y * y}, yz{y * z}, zz{z * z};
    MatNx1 col = { 1, x, y, z, xx, xy, xz, yy, yz, zz,
        x * xx, x * xy, x * xz, x * yy, x * yz, x * zz,
        y * yy, y * yz, y * zz, z * zz };
    return col;
  }
  template <typename MatKxN>
  static MatKxN GetPdvValue(const Coord &xyz, const MatKxN &coeff) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    auto xx{x * x}, xy{x * y}, xz{x * z}, yy{y * y}, yz{y * z}, zz{z * z};
    MatKxN res = coeff; res.col(0).setZero();
    // pdv_x
    assert(res.col(X) == coeff.col(X));
    res.col(X) += coeff.col(XX) * (2 * x);
    res.col(X) += coeff.col(XY) * y;
    res.col(X) += coeff.col(XZ) * z;
    res.col(X) += coeff.col(XXX) * (3 * xx);
    res.col(X) += coeff.col(XXY) * (2 * xy);
    res.col(X) += coeff.col(XXZ) * (2 * xz);
    res.col(X) += coeff.col(XYY) * yy;
    res.col(X) += coeff.col(XYZ) * yz;
    res.col(X) += coeff.col(XZZ) * zz;
    // pdv_y
    assert(res.col(Y) == coeff.col(Y));
    res.col(Y) += coeff.col(XY) * x;
    res.col(Y) += coeff.col(YY) * (2 * y);
    res.col(Y) += coeff.col(YZ) * z;
    res.col(Y) += coeff.col(XXY) * xx;
    res.col(Y) += coeff.col(XYZ) * xz;
    res.col(Y) += coeff.col(XYY) * (2 * xy);
    res.col(Y) += coeff.col(YYY) * (3 * yy);
    res.col(Y) += coeff.col(YYZ) * (2 * yz);
    res.col(Y) += coeff.col(YZZ) * zz;
    // pdv_z
    assert(res.col(Z) == coeff.col(Z));
    res.col(Z) += coeff.col(XZ) * x;
    res.col(Z) += coeff.col(YZ) * y;
    res.col(Z) += coeff.col(ZZ) * (2 * z);
    res.col(Z) += coeff.col(XXZ) * xx;
    res.col(Z) += coeff.col(XYZ) * xy;
    res.col(Z) += coeff.col(XZZ) * (2 * xz);
    res.col(Z) += coeff.col(YYZ) * yy;
    res.col(Z) += coeff.col(YZZ) * (2 * yz);
    res.col(Z) += coeff.col(ZZZ) * (3 * zz);
    // pdv_xx
    res.col(XX) += coeff.col(XXY) * y;
    res.col(XX) += coeff.col(XXZ) * z;
    res.col(XX) += coeff.col(XXX) * x * 3;
    res.col(XX) += res.col(XX);
    assert(res.col(XX) == 2 * (coeff.col(XX) + coeff.col(XXY) * y
        + coeff.col(XXZ) * z + coeff.col(XXX) * x * 3));
    // pdv_xy
    res.col(XY) += coeff.col(XXY) * x * 2;
    res.col(XY) += coeff.col(XYY) * y * 2;
    res.col(XY) += coeff.col(XYZ) * z;
    assert(res.col(XY) == coeff.col(XY) + coeff.col(XYZ) * z
        + 2 * (coeff.col(XXY) * x + coeff.col(XYY) * y));
    // pdv_xz
    res.col(XZ) += coeff.col(XXZ) * x * 2;
    res.col(XZ) += coeff.col(XZZ) * z * 2;
    res.col(XZ) += coeff.col(XYZ) * y;
    assert(res.col(XZ) == coeff.col(XZ) + coeff.col(XYZ) * y
        + 2 * (coeff.col(XXZ) * x + coeff.col(XZZ) * z));
    // pdv_yy
    res.col(YY) += coeff.col(XYY) * x;
    res.col(YY) += coeff.col(YYZ) * z;
    res.col(YY) += coeff.col(YYY) * y * 3;
    res.col(YY) += res.col(XX);
    assert(res.col(YY) == coeff.col(YYY) * y * 6
        + 2 * (coeff.col(YY) + coeff.col(XYY) * x + coeff.col(YYZ) * z));
    // pdv_yz
    res.col(YZ) += coeff.col(XYZ) * x;
    res.col(YZ) += coeff.col(YYZ) * y * 2;
    res.col(YZ) += coeff.col(YZZ) * z * 2;
    assert(res.col(YZ) == coeff.col(YZ) + coeff.col(XYZ) * x
        + 2 * (coeff.col(YYZ) * y + coeff.col(YZZ) * z));
    // pdv_zz
    res.col(ZZ) += coeff.col(XZZ) * x;
    res.col(ZZ) += coeff.col(YZZ) * y;
    res.col(ZZ) += coeff.col(ZZZ) * z * 3;
    res.col(ZZ) += res.col(ZZ);
    assert(res.col(ZZ) == coeff.col(ZZZ) * z * 6
        + 2 * (coeff.col(ZZ) + coeff.col(XZZ) * x + coeff.col(YZZ) * y));
    // pdv_xxx
    res.col(XXX) *= 6;
    assert(res.col(XXX) == coeff.col(XXX) * 6);
    // pdv_xxy
    res.col(XXY) += res.col(XXY);
    assert(res.col(XXY) == coeff.col(XXY) * 2);
    // pdv_xxz
    res.col(XXZ) += res.col(XXZ);
    assert(res.col(XXZ) == coeff.col(XXZ) * 2);
    // pdv_xyy
    res.col(XYY) += res.col(XYY);
    assert(res.col(XYY) == coeff.col(XYY) * 2);
    // pdv_xyz
    assert(res.col(XYZ) == coeff.col(XYZ));
    // pdv_xzz
    res.col(XZZ) += res.col(XZZ);
    assert(res.col(XZZ) == coeff.col(XZZ) * 2);
    // pdv_yyy
    res.col(YYY) *= 6;
    assert(res.col(YYY) == coeff.col(YYY) * 6);
    // pdv_yyz
    res.col(YYZ) += res.col(YYZ);
    assert(res.col(YYZ) == coeff.col(YYZ) * 2);
    // pdv_yzz
    res.col(YZZ) += res.col(YZZ);
    assert(res.col(YZZ) == coeff.col(YZZ) * 2);
    // pdv_zzz
    res.col(ZZZ) *= 6;
    assert(res.col(ZZZ) == coeff.col(ZZZ) * 2);
    return res;
  }

  template <int K>
  static auto GetGradValue(const Coord &xyz,
      const algebra::Matrix<Scalar, K, N> &coeff) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    auto xx{x * x}, xy{x * y}, xz{x * z}, yy{y * y}, yz{y * z}, zz{z * z};
    algebra::Matrix<Scalar, K, 3> res;
    int i = 0;
    // pdv_x
    res.col(i) = coeff.col(X);
    res.col(i) += coeff.col(XX) * (2 * x);
    res.col(i) += coeff.col(XY) * y;
    res.col(i) += coeff.col(XZ) * z;
    res.col(i) += coeff.col(XXX) * (3 * xx);
    res.col(i) += coeff.col(XXY) * (2 * xy);
    res.col(i) += coeff.col(XXZ) * (2 * xz);
    res.col(i) += coeff.col(XYY) * yy;
    res.col(i) += coeff.col(XYZ) * yz;
    res.col(i) += coeff.col(XZZ) * zz;
    // pdv_y
    i++;
    res.col(i) = coeff.col(Y);
    res.col(i) += coeff.col(XY) * x;
    res.col(i) += coeff.col(YY) * (2 * y);
    res.col(i) += coeff.col(YZ) * z;
    res.col(i) += coeff.col(XXY) * xx;
    res.col(i) += coeff.col(XYZ) * xz;
    res.col(i) += coeff.col(XYY) * (2 * xy);
    res.col(i) += coeff.col(YYY) * (3 * yy);
    res.col(i) += coeff.col(YYZ) * (2 * yz);
    res.col(i) += coeff.col(YZZ) * zz;
    // pdv_z
    i++;
    res.col(i) = coeff.col(Z);
    res.col(i) += coeff.col(XZ) * x;
    res.col(i) += coeff.col(YZ) * y;
    res.col(i) += coeff.col(ZZ) * (2 * z);
    res.col(i) += coeff.col(XXZ) * xx;
    res.col(i) += coeff.col(XYZ) * xy;
    res.col(i) += coeff.col(XZZ) * (2 * xz);
    res.col(i) += coeff.col(YYZ) * yy;
    res.col(i) += coeff.col(YZZ) * (2 * yz);
    res.col(i) += coeff.col(ZZZ) * (3 * zz);
    return res;
  }

  template <int K>
  static auto GetSmoothness(
      const algebra::Matrix<Scalar, K, N> &integral, Scalar volume) {
    using MatKx1 = algebra::Matrix<Scalar, K, 1>;
    auto w1  // weight of 1st-order partial derivatives
        = std::pow(volume, 2./3-1);
    MatKx1 smoothness = integral.col(X);
    smoothness += integral.col(Y);
    smoothness += integral.col(Z);
    smoothness *= w1;
    auto w2  // weight of 2nd-order partial derivatives
        = std::pow(volume, 4./3-1);
    smoothness += integral.col(XX) * w2;
    smoothness += integral.col(XY) * w2;
    smoothness += integral.col(XZ) * w2;
    smoothness += integral.col(YY) * w2;
    smoothness += integral.col(YZ) * w2;
    smoothness += integral.col(ZZ) * w2;
    auto w3  // weight of 3rd-order partial derivatives
        = volume;  // = std::pow(volume, 6./3-1);
    smoothness += integral.col(XXX) * w3;
    smoothness += integral.col(XXY) * w3;
    smoothness += integral.col(XXZ) * w3;
    smoothness += integral.col(XYY) * w3;
    smoothness += integral.col(XYZ) * w3;
    smoothness += integral.col(XZZ) * w3;
    smoothness += integral.col(YYY) * w3;
    smoothness += integral.col(YYZ) * w3;
    smoothness += integral.col(YZZ) * w3;
    smoothness += integral.col(ZZZ) * w3;
    return smoothness;
  }
};

}  // namespace basis
}  // namespace mini

#endif  // MINI_BASIS_TAYLOR_HPP_
