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

namespace {

template <int D>
constexpr int GetN(int P);

template <>
constexpr int GetN<1>(int P) {
  return 1 + P;
}

template <int D>
constexpr int GetN(int P) {
  return (GetN<D - 1>(P) * (D + P)) / D;
}

template <int kDimensions>
struct _Index;

template <>
struct _Index<1> {
  enum {
    X = 1, XX, XXX, XXXX, XXXXX, XXXXXX,
  };
};

template <>
struct _Index<2> {
  enum {
    X = 1, Y,
    XX, XY, YY,
    XXX, XXY, XYY, YYY,
    XXXX, XXXY, XXYY, XYYY, YYYY,
    XXXXX,
  };
  static_assert(XXXXX == GetN<2>(4));
};

template <>
struct _Index<3> {
  enum {
    X = 1, Y, Z,
    XX, XY, XZ,
        YY, YZ, ZZ,
    XXX, XXY, XXZ, XYY, XYZ, XZZ,
                   YYY, YYZ, YZZ, ZZZ,
    XXXX, XXXY, XXXZ, XXYY, XXYZ, XXZZ, XYYY, XYYZ, XYZZ, XZZZ,
                                        YYYY, YYYZ, YYZZ, YZZZ, ZZZZ,
    XXXXX,
  };
  static_assert(XXXXX == GetN<3>(4));
};

}

/**
 * @brief The basis, formed by monomials, of the space spanned by polynomials.
 * 
 * @tparam S the type of coordinates
 * @tparam kDimensions the dimension of underlying space
 * @tparam kDegrees the maximum degree of its members
 */
template <std::floating_point S, int kDimensions, int kDegrees>
class Taylor;

template <std::floating_point S, int kDegrees>
class Taylor<S, 1, kDegrees> {
 public:
  static constexpr int D = 1;

  // the maximum degree of members in this basis
  static constexpr int P = kDegrees;

  // the number of terms in this basis
  static constexpr int N = GetN<D>(P);

  using Scalar = S;
  using Vector = algebra::Vector<Scalar, N>;

  using Index = _Index<1>;

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

template <std::floating_point S, int kDimensions, int kDegrees>
class Taylor {
 public:
  static constexpr int P = kDegrees;
  static constexpr int D = kDimensions;
  static constexpr int N = GetN<D>(P);  // the number of components
  using Scalar = S;

  template <int kRows, int kColumns>
  using Matrix = algebra::Matrix<Scalar, kRows, kColumns>;

  using MatNx1 = Matrix<N, 1>;
  using Coord = Matrix<D, 1>;

  using Index = _Index<kDimensions>;

 public:
  static MatNx1 GetValue(const Coord &coord) {
    return _GetValue(coord);
  }

 private:
  static MatNx1 _GetValue(const Coord &xy) requires(kDegrees == 0) {
    return { 1 };
  }

  static MatNx1 _GetValue(const Coord &xy)
      requires(kDimensions == 2 && kDegrees == 1) {
    return { 1, xy[0], xy[1] };
  }

  static MatNx1 _GetValue(const Coord &xy)
      requires(kDimensions == 2 && kDegrees == 2) {
    auto x = xy[0], y = xy[1];
    return { 1, x, y, x * x, x * y, y * y };
  }

  static MatNx1 _GetValue(const Coord &xy)
      requires(kDimensions == 2 && kDegrees == 3) {
    auto x = xy[0], y = xy[1];
    auto x_x = x * x, x_y = x * y, y_y = y * y;
    return { 1, x, y, x_x, x_y, y_y,
        x_x * x, x_x * y, x * y_y, y * y_y };
  }

  static MatNx1 _GetValue(const Coord &xyz)
      requires(kDimensions == 3 && kDegrees == 1) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    return { 1, x, y, z };
  }

  static MatNx1 _GetValue(const Coord &xyz)
      requires(kDimensions == 3 && kDegrees == 2) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    return { 1, x, y, z, x * x, x * y, x * z, y * y, y * z, z * z };
  }

  static MatNx1 _GetValue(const Coord &xyz)
      requires(kDimensions == 3 && kDegrees == 3) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    auto xx{x * x}, xy{x * y}, xz{x * z}, yy{y * y}, yz{y * z}, zz{z * z};
    return { 1, x, y, z, xx, xy, xz, yy, yz, zz,
        x * xx, x * xy, x * xz, x * yy, x * yz, x * zz,
        y * yy, y * yz, y * zz, z * zz };
  }

 public:
  template <int K> static Matrix<K, N>
  GetPartialDerivatives(const Coord &xyz, const Matrix<K, N> &coeff)
      requires(2 <= D && D <= 3 && 0 <= P && P <= 3) {
    return _GetPartialDerivatives(xyz, coeff);
  }

 private:
  template <int K> static Matrix<K, N>
  _GetPartialDerivatives(const Coord &xyz, const Matrix<K, N> &coeff)
      requires(P == 0) {
    return Matrix<K, N>::Zero();
  }

  template <int K> static Matrix<K, N>
  _GetPartialDerivatives(const Coord &xyz, const Matrix<K, N> &coeff)
      requires(P == 1) {
    Matrix<K, N> res = coeff;
    res.col(0).setZero();
    return res;
  }

  template <int K> static Matrix<K, N>
  _GetPartialDerivatives(const Coord &xyz, const Matrix<K, N> &coeff)
      requires(D == 3 && P == 2) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    Matrix<K, N> res = coeff;
    res.col(0).setZero();
    // pdv_x
    assert(res.col(Index::X) == coeff.col(Index::X));
    res.col(Index::X) += coeff.col(Index::XX) * (2 * x);
    res.col(Index::X) += coeff.col(Index::XY) * y;
    res.col(Index::X) += coeff.col(Index::XZ) * z;
    // pdv_y
    assert(res.col(Index::Y) == coeff.col(Index::Y));
    res.col(Index::Y) += coeff.col(Index::XY) * x;
    res.col(Index::Y) += coeff.col(Index::YY) * (2 * y);
    res.col(Index::Y) += coeff.col(Index::YZ) * z;
    // pdv_z
    assert(res.col(Index::Z) == coeff.col(Index::Z));
    res.col(Index::Z) += coeff.col(Index::XZ) * x;
    res.col(Index::Z) += coeff.col(Index::YZ) * y;
    res.col(Index::Z) += coeff.col(Index::ZZ) * (2 * z);
    // pdv_xx
    res.col(Index::XX) += coeff.col(Index::XX);
    assert(res.col(Index::XX) == coeff.col(Index::XX) * 2);
    // pdv_xy
    assert(res.col(Index::XY) == coeff.col(Index::XY));
    // pdv_xz
    assert(res.col(Index::XZ) == coeff.col(Index::XZ));
    // pdv_yy
    res.col(Index::YY) += coeff.col(Index::YY);
    assert(res.col(Index::YY) == coeff.col(Index::YY) * 2);
    // pdv_yz
    assert(res.col(Index::YZ) == coeff.col(Index::YZ));
    // pdv_zz
    res.col(Index::ZZ) += coeff.col(Index::ZZ);
    assert(res.col(Index::ZZ) == coeff.col(Index::ZZ) * 2);
    return res;
  }

  template <int K> static Matrix<K, N>
  _GetPartialDerivatives(const Coord &xyz, const Matrix<K, N> &coeff)
      requires(D == 3 && P == 3) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    auto xx{x * x}, xy{x * y}, xz{x * z}, yy{y * y}, yz{y * z}, zz{z * z};
    Matrix<K, N> res = coeff;
    res.col(0).setZero();
    // pdv_x
    assert(res.col(Index::X) == coeff.col(Index::X));
    res.col(Index::X) += coeff.col(Index::XX) * (2 * x);
    res.col(Index::X) += coeff.col(Index::XY) * y;
    res.col(Index::X) += coeff.col(Index::XZ) * z;
    res.col(Index::X) += coeff.col(Index::XXX) * (3 * xx);
    res.col(Index::X) += coeff.col(Index::XXY) * (2 * xy);
    res.col(Index::X) += coeff.col(Index::XXZ) * (2 * xz);
    res.col(Index::X) += coeff.col(Index::XYY) * yy;
    res.col(Index::X) += coeff.col(Index::XYZ) * yz;
    res.col(Index::X) += coeff.col(Index::XZZ) * zz;
    // pdv_y
    assert(res.col(Index::Y) == coeff.col(Index::Y));
    res.col(Index::Y) += coeff.col(Index::XY) * x;
    res.col(Index::Y) += coeff.col(Index::YY) * (2 * y);
    res.col(Index::Y) += coeff.col(Index::YZ) * z;
    res.col(Index::Y) += coeff.col(Index::XXY) * xx;
    res.col(Index::Y) += coeff.col(Index::XYZ) * xz;
    res.col(Index::Y) += coeff.col(Index::XYY) * (2 * xy);
    res.col(Index::Y) += coeff.col(Index::YYY) * (3 * yy);
    res.col(Index::Y) += coeff.col(Index::YYZ) * (2 * yz);
    res.col(Index::Y) += coeff.col(Index::YZZ) * zz;
    // pdv_z
    assert(res.col(Index::Z) == coeff.col(Index::Z));
    res.col(Index::Z) += coeff.col(Index::XZ) * x;
    res.col(Index::Z) += coeff.col(Index::YZ) * y;
    res.col(Index::Z) += coeff.col(Index::ZZ) * (2 * z);
    res.col(Index::Z) += coeff.col(Index::XXZ) * xx;
    res.col(Index::Z) += coeff.col(Index::XYZ) * xy;
    res.col(Index::Z) += coeff.col(Index::XZZ) * (2 * xz);
    res.col(Index::Z) += coeff.col(Index::YYZ) * yy;
    res.col(Index::Z) += coeff.col(Index::YZZ) * (2 * yz);
    res.col(Index::Z) += coeff.col(Index::ZZZ) * (3 * zz);
    // pdv_xx
    res.col(Index::XX) += coeff.col(Index::XXY) * y;
    res.col(Index::XX) += coeff.col(Index::XXZ) * z;
    res.col(Index::XX) += coeff.col(Index::XXX) * x * 3;
    res.col(Index::XX) += res.col(Index::XX);
    // pdv_xy
    res.col(Index::XY) += coeff.col(Index::XXY) * x * 2;
    res.col(Index::XY) += coeff.col(Index::XYY) * y * 2;
    res.col(Index::XY) += coeff.col(Index::XYZ) * z;
    // pdv_xz
    res.col(Index::XZ) += coeff.col(Index::XXZ) * x * 2;
    res.col(Index::XZ) += coeff.col(Index::XZZ) * z * 2;
    res.col(Index::XZ) += coeff.col(Index::XYZ) * y;
    // pdv_yy
    res.col(Index::YY) += coeff.col(Index::XYY) * x;
    res.col(Index::YY) += coeff.col(Index::YYZ) * z;
    res.col(Index::YY) += coeff.col(Index::YYY) * y * 3;
    res.col(Index::YY) += res.col(Index::YY);
    // pdv_yz
    res.col(Index::YZ) += coeff.col(Index::XYZ) * x;
    res.col(Index::YZ) += coeff.col(Index::YYZ) * y * 2;
    res.col(Index::YZ) += coeff.col(Index::YZZ) * z * 2;
    // pdv_zz
    res.col(Index::ZZ) += coeff.col(Index::XZZ) * x;
    res.col(Index::ZZ) += coeff.col(Index::YZZ) * y;
    res.col(Index::ZZ) += coeff.col(Index::ZZZ) * z * 3;
    res.col(Index::ZZ) += res.col(Index::ZZ);
    // pdv_xxx
    res.col(Index::XXX) *= 6;
    assert(res.col(Index::XXX) == coeff.col(Index::XXX) * 6);
    // pdv_xxy
    res.col(Index::XXY) += res.col(Index::XXY);
    assert(res.col(Index::XXY) == coeff.col(Index::XXY) * 2);
    // pdv_xxz
    res.col(Index::XXZ) += res.col(Index::XXZ);
    assert(res.col(Index::XXZ) == coeff.col(Index::XXZ) * 2);
    // pdv_xyy
    res.col(Index::XYY) += res.col(Index::XYY);
    assert(res.col(Index::XYY) == coeff.col(Index::XYY) * 2);
    // pdv_xyz
    assert(res.col(Index::XYZ) == coeff.col(Index::XYZ));
    // pdv_xzz
    res.col(Index::XZZ) += res.col(Index::XZZ);
    assert(res.col(Index::XZZ) == coeff.col(Index::XZZ) * 2);
    // pdv_yyy
    res.col(Index::YYY) *= 6;
    assert(res.col(Index::YYY) == coeff.col(Index::YYY) * 6);
    // pdv_yyz
    res.col(Index::YYZ) += res.col(Index::YYZ);
    assert(res.col(Index::YYZ) == coeff.col(Index::YYZ) * 2);
    // pdv_yzz
    res.col(Index::YZZ) += res.col(Index::YZZ);
    assert(res.col(Index::YZZ) == coeff.col(Index::YZZ) * 2);
    // pdv_zzz
    res.col(Index::ZZZ) *= 6;
    assert(res.col(Index::ZZZ) == coeff.col(Index::ZZZ) * 6);
    return res;
  }

 public:
  template <int K> static Matrix<K, D>
  GetGradValue(const Coord &xyz, const Matrix<K, N> &coeff)
      requires(2 <= D && D <= 3 && 0 <= P && P <= 3) {
    return _GetGradValue(xyz, coeff); 
  }

 private:
  template <int K> static Matrix<K, D>
  _GetGradValue(const Coord &xyz, const Matrix<K, N> &coeff)
      requires(P == 0) {
    return Matrix<K, D>::Zero();
  }

  template <int K> static Matrix<K, D>
  _GetGradValue(const Coord &xyz, const Matrix<K, N> &coeff)
        requires(D == 3 && P == 1) {
    Matrix<K, D> res;
    // pdv_x
    res.col(0) = coeff.col(Index::X);
    // pdv_y
    res.col(1) = coeff.col(Index::Y);
    // pdv_z
    res.col(2) = coeff.col(Index::Z);
    return res;
  }

  template <int K> static Matrix<K, D>
  _GetGradValue(const Coord &xyz, const Matrix<K, N> &coeff)
      requires(D == 3 && P == 2) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    Matrix<K, D> res;
    // pdv_x
    res.col(0) = coeff.col(Index::X);
    res.col(0) += coeff.col(Index::XX) * (2 * x);
    res.col(0) += coeff.col(Index::XY) * y;
    res.col(0) += coeff.col(Index::XZ) * z;
    // pdv_y
    res.col(1) = coeff.col(Index::Y);
    res.col(1) += coeff.col(Index::XY) * x;
    res.col(1) += coeff.col(Index::YY) * (2 * y);
    res.col(1) += coeff.col(Index::YZ) * z;
    // pdv_z
    res.col(2) = coeff.col(Index::Z);
    res.col(2) += coeff.col(Index::XZ) * x;
    res.col(2) += coeff.col(Index::YZ) * y;
    res.col(2) += coeff.col(Index::ZZ) * (2 * z);
    return res;
  }

  template <int K> static Matrix<K, D>
  _GetGradValue(const Coord &xyz, const Matrix<K, N> &coeff)
        requires(D == 3 && P == 3) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    auto xx{x * x}, xy{x * y}, xz{x * z}, yy{y * y}, yz{y * z}, zz{z * z};
    Matrix<K, D> res;
    // pdv_x
    res.col(0) = coeff.col(Index::X);
    res.col(0) += coeff.col(Index::XX) * (2 * x);
    res.col(0) += coeff.col(Index::XY) * y;
    res.col(0) += coeff.col(Index::XZ) * z;
    res.col(0) += coeff.col(Index::XXX) * (3 * xx);
    res.col(0) += coeff.col(Index::XXY) * (2 * xy);
    res.col(0) += coeff.col(Index::XXZ) * (2 * xz);
    res.col(0) += coeff.col(Index::XYY) * yy;
    res.col(0) += coeff.col(Index::XYZ) * yz;
    res.col(0) += coeff.col(Index::XZZ) * zz;
    // pdv_y
    res.col(1) = coeff.col(Index::Y);
    res.col(1) += coeff.col(Index::XY) * x;
    res.col(1) += coeff.col(Index::YY) * (2 * y);
    res.col(1) += coeff.col(Index::YZ) * z;
    res.col(1) += coeff.col(Index::XXY) * xx;
    res.col(1) += coeff.col(Index::XYZ) * xz;
    res.col(1) += coeff.col(Index::XYY) * (2 * xy);
    res.col(1) += coeff.col(Index::YYY) * (3 * yy);
    res.col(1) += coeff.col(Index::YYZ) * (2 * yz);
    res.col(1) += coeff.col(Index::YZZ) * zz;
    // pdv_z
    res.col(2) = coeff.col(Index::Z);
    res.col(2) += coeff.col(Index::XZ) * x;
    res.col(2) += coeff.col(Index::YZ) * y;
    res.col(2) += coeff.col(Index::ZZ) * (2 * z);
    res.col(2) += coeff.col(Index::XXZ) * xx;
    res.col(2) += coeff.col(Index::XYZ) * xy;
    res.col(2) += coeff.col(Index::XZZ) * (2 * xz);
    res.col(2) += coeff.col(Index::YYZ) * yy;
    res.col(2) += coeff.col(Index::YZZ) * (2 * yz);
    res.col(2) += coeff.col(Index::ZZZ) * (3 * zz);
    return res;
  }
};

}  // namespace basis
}  // namespace mini

#endif  // MINI_BASIS_TAYLOR_HPP_
