//  Copyright 2023 PEI Weicheng
#ifndef MINI_BASIS_LAGRANGE_HPP_
#define MINI_BASIS_LAGRANGE_HPP_

#include <concepts>
#include <ranges>

#include <cassert>
#include <cmath>

#include <algorithm>
#include <array>
#include <initializer_list>
#include <tuple>
#include <utility>

#include "mini/algebra/eigen.hpp"

#include "mini/basis/taylor.hpp"


namespace mini {
namespace basis {
namespace lagrange {

/**
 * @brief The Lagrange basis functions on the standard Line element.
 * 
 * The Lagrange polynomal at the \f$k\f$-th node \f$ L_k(\xi) \f$ is defined as \f$ \prod_{l=0,l\ne k}^{N-1}\frac{\xi-\xi_{l}}{\xi_{k}-\xi_{l}} \f$.
 * 
 * @tparam S the type of nodes
 * @tparam kDegree the degree of each Lagrange polynomial
 */
template <std::floating_point S, int kDegree>
class Line {
 public:
  static constexpr int P = kDegree;  // the degree of each member in this basis
  static constexpr int N = P + 1;  // the number of terms in this basis
  using Scalar = S;
  using Vector = algebra::Matrix<Scalar, 1, N>;
  using Matrix = algebra::Matrix<Scalar, N, N>;
  using Taylor = basis::Taylor<Scalar, 1, kDegree>;

 private:
  /**
   * derivatives_[k][j][i] := k-th order derivatives at the j-th node of the i-th basis
   */
  std::array<std::array<Vector, N>, N> derivatives_;
  Vector nodes_;  // nodes_[j] := coorinate of the j-th node
  Matrix lagrange_to_taylor_;

 public:
  template <std::ranges::random_access_range R>
  explicit Line(R &&nodes) {
    assert(nodes.size() == N);
    /* An arbitrary polynomial can be expanded on Taylor and Lagrange:
     *     polynomial(x) = taylor_basis_row(x) @ taylor_coeff_col
     *                   = lagrange_basis_row(x) @ lagrange_coeff_col.
     * We want to find `taylor_to_lagrange` and `lagrange_to_taylor` such that
     *     taylor_coeff_col = lagrange_to_taylor @ lagrange_coeff_col,
     * and lagrange_coeff_col = taylor_to_lagrange @ taylor_coeff_col.
     * Then, from
     *     taylor_basis_row(x) @             taylor_coeff_col            = 
     *     taylor_basis_row(x) @ lagrange_to_taylor @ lagrange_coeff_col =
     *                   lagrange_basis_row(x)      @ lagrange_coeff_col,
     * We have
     *     taylor_basis_row(x) @ lagrange_to_taylor = lagrange_basis_row(x),
     * and lagrange_basis_row(x) @ taylor_to_lagrange = taylor_basis_row(x).
     * Since lagrange_basis_row(x[j])[i] = delta[i][j], we have
     *     taylor_to_lagrange[j] := values of Taylor basis at the j-th node.
     */
    Matrix taylor_to_lagrange;
    for (int j = 0; j < N; ++j) {
      auto x_j = (nodes_[j] = nodes[j]);
      taylor_to_lagrange.row(j) = Taylor::GetValues(x_j);
    }
    lagrange_to_taylor_ = taylor_to_lagrange.inverse();
    assert((taylor_to_lagrange * lagrange_to_taylor_
        - Matrix::Identity()).norm() < 1e-12);
    for (int k = 0; k < N; ++k) {
      for (int j = 0; j < N; ++j) {
        auto x_j = nodes_[j];
        derivatives_[k][j] = GetDerivatives(k, x_j);
      }
    }
  }

  explicit Line(std::initializer_list<Scalar> nodes)
      : Line(std::views::iota(nodes.begin(), nodes.end())
           | std::views::transform([](auto *ptr){ return *ptr; })) {
  }

  /**
   * @brief Get the coordinate of the \f$i\f$-th node.
   * 
   * @param i the (0-based) index of the query node
   * @return Scalar the coordinate
   */
  Scalar GetNode(int i) const {
    return nodes_[i];
  }

  /**
   * @brief Get the values of all basis functions at an arbitrary point.
   * 
   * @param x the coordinate of the query point
   * @return Vector the values
   */
  Vector GetValues(Scalar x) const {
    Vector vec; vec.setOnes();
    for (int i = 0; i < N; ++i) {
      auto x_i = nodes_[i];
      for (int j = 0; j < N; ++j) {
        if (i != j) {
          auto x_j = nodes_[j];
          vec[i] *= (x - x_j) / (x_i - x_j);
        }
      }
    }
    return vec;
  }

  /**
   * @brief Get the \f$k\f$-th order derivatives of all basis functions at an arbitrary point.
   * 
   * @param k the order of the derivatives to be taken
   * @param x the coordinate of the query point
   * @return Vector the derivatives
   */
  Vector GetDerivatives(int k, Scalar x) const {
    Vector taylor_basis_row = Taylor::GetDerivatives(k, x);
    return taylor_basis_row * lagrange_to_taylor_;
  }

  /**
   * @brief Get the \f$k\f$-th order derivatives of all basis functions at the \f$j\f$-th node.
   * 
   * @param k the order of the derivatives to be taken
   * @param j the (0-based) index of the query node
   * @return Vector the derivatives
   */
  Vector const &GetDerivatives(int k, int j) const {
    return derivatives_[k][j];
  }
};

/**
 * @brief The Lagrange basis functions on the standard Hexahedron element.
 * 
 * The Lagrange polynomal at the \f$(i,j,k)\f$-th node \f$ L_{i,j,k}(\xi, \eta, \zeta) \f$ is defined as \f$ L_i(\xi)\,L_j(\eta)\,L_k(\zeta) \f$.
 * 
 * @tparam S the type of coordinates
 * @tparam kDegreeX the degree of each \f$ L_i(\xi) \f$
 * @tparam kDegreeY the degree of each \f$ L_j(\eta) \f$
 * @tparam kDegreeZ the degree of each \f$ L_k(\zeta) \f$
 */
template <std::floating_point S, int kDegreeX, int kDegreeY, int kDegreeZ>
class Hexahedron {
 public:
  using Scalar = S;

  // the type of the Lagrange basis in the 1st dimension
  using LineX = Line<Scalar, kDegreeX>;

  // the type of the Lagrange basis in the 2nd dimension
  using LineY = Line<Scalar, kDegreeY>;

  // the type of the Lagrange basis in the 3rd dimension
  using LineZ = Line<Scalar, kDegreeZ>;

  // the number of terms in the 1st dimension
  static constexpr int I = LineX::N;

  // the number of terms in the 2nd dimension
  static constexpr int J = LineY::N;

  // the number of terms in the 3rd dimension
  static constexpr int K = LineZ::N;

  // the number of terms in this basis
  static constexpr int N = I * J * K;

  // the 1D type of values of this basis
  using Vector = algebra::Matrix<Scalar, 1, N>;

  // the type of coordinates
  using Coord = algebra::Vector<Scalar, 3>;

 private:
  LineX line_x_;
  LineY line_y_;
  LineZ line_z_;
  Vector derivatives_[I][J][K][I][J][K];

 public:
  Hexahedron(LineX const &line_x, LineY const &line_y, LineZ const &line_z)
      : line_x_(line_x), line_y_(line_y), line_z_(line_z) {
    // cache derivatives at nodes
    for (int a = 0; a < I; ++a) {
      for (int b = 0; b < J; ++b) {
        for (int c = 0; c < K; ++c) {
          for (int i = 0; i < I; ++i) {
            auto x = line_x_.GetNode(i);
            for (int j = 0; j < J; ++j) {
              auto y = line_y_.GetNode(j);
              for (int k = 0; k < K; ++k) {
                auto z = line_z_.GetNode(k);
                derivatives_[a][b][c][i][j][k] =
                    GetDerivatives(a, b, c, x, y, z);
              }
            }
          }
        }
      }
    }
  }

  /**
   * @brief Get the coordinate of the \f$(i,j,k)\f$-th node.
   * 
   * @param i the (0-based) index of the query node in the 1st dimension
   * @param j the (0-based) index of the query node in the 2nd dimension
   * @param k the (0-based) index of the query node in the 3rd dimension
   * @return Scalar the coordinate
   */
  Coord GetNode(int i, int j, int k) const {
    Coord coord{ line_x_.GetNode(i), line_y_.GetNode(j), line_z_.GetNode(k) };
    return coord;
  }

  /**
   * @brief Get the 1D index from the 3D index.
   * 
   * @param i the (0-based) index in the 1st dimension
   * @param j the (0-based) index in the 2nd dimension
   * @param k the (0-based) index in the 3rd dimension
   * @return int the 1D index
   */
  static int index(int i, int j, int k) {
    return i * J * K + j * K + k;
  }
  /**
   * @brief Get the 3D index from the 1D index.
   * 
   * @param ijk the (0-based) 1D index
   * @return int the 1D index
   */
  static std::tuple<int, int, int> index(int ijk) {
    int k = ijk % K;
    int j = ijk / K;  // i * J + j
    int i = j / J; j %= J;
    return std::make_tuple(i, j, k);
  }

  /**
   * @brief Get the coordinate of the \f$ijk\f$-th node.
   * 
   * @param ijk the (0-based) 1D index of the query node
   * @return Scalar the coordinate
   */
  Coord GetNode(int ijk) const {
    auto [i, j, k] = index(ijk);
    assert(index(i, j, k) == ijk);
    return GetNode(i, j, k);
  }

  /**
   * @brief Get the values of all basis functions at an arbitrary point.
   * 
   * @param x the coordinate of the query point in the 1st dimension
   * @param y the coordinate of the query point in the 2nd dimension
   * @param z the coordinate of the query point in the 3rd dimension
   * @return Vector the output values
   */
  Vector GetValues(Scalar x, Scalar y, Scalar z) const {
    Vector vec;
    auto value_x = line_x_.GetValues(x);
    auto value_y = line_y_.GetValues(y);
    auto value_z = line_z_.GetValues(z);
    int ijk = 0;
    for (int i = 0; i < I; ++i) {
      for (int j = 0; j < J; ++j) {
        auto value_xy = value_x[i] * value_y[j];
        for (int k = 0; k < K; ++k) {
          vec[ijk++] = value_xy * value_z[k];
        }
      }
    }
    assert(ijk == N);
    return vec;
  }

  /**
   * @brief Get the values of all basis functions at an arbitrary point.
   * 
   * @param coord the coordinates of the query point
   * @return Vector the output values
   */
  Vector GetValues(Coord const &coord) const {
    return GetValues(coord[0], coord[1], coord[2]);
  }

  /**
   * @brief Get the \f$(a,b,c)\f$-th order derivatives of all basis functions at an arbitrary point.
   * 
   * @param a the order of the derivatives to be taken in the 1st dimension
   * @param b the order of the derivatives to be taken in the 1st dimension
   * @param c the order of the derivatives to be taken in the 1st dimension
   * @param x the coordinate of the query point in the 1st dimension
   * @param y the coordinate of the query point in the 2nd dimension
   * @param z the coordinate of the query point in the 3rd dimension
   * @return Vector the output derivatives
   */
  Vector GetDerivatives(
      int a, int b, int c, Scalar x, Scalar y, Scalar z) const {
    Vector vec;
    auto value_x = line_x_.GetDerivatives(a, x);
    auto value_y = line_y_.GetDerivatives(b, y);
    auto value_z = line_z_.GetDerivatives(c, z);
    int ijk = 0;
    for (int i = 0; i < I; ++i) {
      for (int j = 0; j < J; ++j) {
        auto value_xy = value_x[i] * value_y[j];
        for (int k = 0; k < K; ++k) {
          vec[ijk++] = value_xy * value_z[k];
        }
      }
    }
    assert(ijk == N);
    return vec;
  }

  /**
   * @brief Get the \f$(a,b,c)\f$-th order derivatives of all basis functions at an arbitrary point.
   * 
   * @param a the order of the derivatives to be taken in the 1st dimension
   * @param b the order of the derivatives to be taken in the 1st dimension
   * @param c the order of the derivatives to be taken in the 1st dimension
   * @param coord the coordinate of the query point
   * @return Vector the output derivatives
   */
  Vector GetDerivatives(
      int a, int b, int c, const Coord &coord) const {
    return GetDerivatives(a, b, c, coord[0], coord[1], coord[2]);
  }

  /**
   * @brief Get the \f$(a,b,c)\f$-th order derivatives of all basis functions at the \f$(i,j,k)\f$-th node.
   * 
   * @param a the order of the derivatives to be taken in the 1st dimension
   * @param b the order of the derivatives to be taken in the 1st dimension
   * @param c the order of the derivatives to be taken in the 1st dimension
   * @param i the (0-based) index of the query node in the 1st dimension
   * @param j the (0-based) index of the query node in the 2nd dimension
   * @param k the (0-based) index of the query node in the 3rd dimension
   * @return Vector const& the derivatives
   */
  Vector const &GetDerivatives(int a, int b, int c, int i, int j, int k) const {
    return derivatives_[a][b][c][i][j][k];
  }
};

}  // namespace lagrange
}  // namespace basis
}  // namespace mini

#endif  // MINI_BASIS_LAGRANGE_HPP_
