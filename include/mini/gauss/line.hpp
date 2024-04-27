//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_GAUSS_LINE_HPP_
#define MINI_GAUSS_LINE_HPP_

#include <cmath>
#include <type_traits>

#include "mini/algebra/eigen.hpp"
#include "mini/gauss/legendre.hpp"
#include "mini/gauss/lobatto.hpp"

namespace mini {
namespace gauss {

enum class Rule { kLegendre, kLobatto };

template <typename S, int D, int Q>
class Line {
 private:
  using MatDx1 = algebra::Matrix<S, D, 1>;

 public:
  using Scalar = S;
  using Local = Scalar;
  using Global = std::conditional_t<(D > 1), MatDx1, Scalar>;

 private:
  using MatDx2 = algebra::Vector<Global, 2>;
  using Mat2x1 = algebra::Vector<Scalar, 2>;
  using Gauss = Legendre<Scalar, Q>;

 private:
  static const std::array<Scalar, Q> local_weights_;
  static const std::array<Local, Q> local_coords_;
  std::array<Scalar, Q> global_weights_;
  std::array<Global, Q> global_coords_;
  MatDx2 pq_;

 public:
  static int CountCorners() {
    return 2;
  }
  static int CountPoints() {
    return Q;
  }
  Global GetVertex(int i) const {
    return pq_[i];
  }

 private:
  static Scalar Norm(Scalar v) {
    return std::abs(v);
  }
  static Scalar Norm(MatDx1 const &v) {
    return v.norm();
  }
  void BuildQuadraturePoints() {
    Global pq = pq_[0] - pq_[1];
    auto det_j = Norm(pq) * 0.5;
    int n = CountPoints();
    for (int i = 0; i < n; ++i) {
      global_weights_[i] = local_weights_[i] * det_j;
      global_coords_[i] = LocalToGlobal(GetLocal(i));
    }
  }
  static constexpr auto BuildLocalCoords() {
    return Gauss::points;
  }
  static constexpr auto BuildLocalWeights() {
    return Gauss::weights;
  }
  static Mat2x1 shape_2x1(Scalar x) {
    x *= 0.5;
    return { 0.5 - x, 0.5 + x };
  }

 public:
  Line(Global const &p0, Global const &p1) {
    pq_[0] = p0; pq_[1] = p1;
    BuildQuadraturePoints();
  }
  Global const &GetGlobal(int i) const {
    assert(0 <= i && i < CountPoints());
    return global_coords_[i];
  }
  Scalar const &GetGlobalWeight(int i) const {
    assert(0 <= i && i < CountPoints());
    return global_weights_[i];
  }
  Local const &GetLocal(int i) const {
    assert(0 <= i && i < CountPoints());
    return local_coords_[i];
  }
  Scalar const &GetLocalWeight(int i) const {
    assert(0 <= i && i < CountPoints());
    return local_weights_[i];
  }
  Global LocalToGlobal(Scalar x) const {
    return pq_.dot(shape_2x1(x));
  }
};

template <typename S, int D, int Q>
std::array<typename Line<S, D, Q>::Local, Q> const
Line<S, D, Q>::local_coords_ = Line<S, D, Q>::BuildLocalCoords();

template <typename S, int D, int Q>
std::array<S, Q> const
Line<S, D, Q>::local_weights_ = Line<S, D, Q>::BuildLocalWeights();

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_LINE_HPP_
