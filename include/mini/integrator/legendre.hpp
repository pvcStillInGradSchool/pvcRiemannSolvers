//  Copyright 2019 PEI Weicheng and YANG Minghao
#ifndef MINI_INTEGRATOR_LEGENDRE_HPP_
#define MINI_INTEGRATOR_LEGENDRE_HPP_

#include <concepts>

#include <cmath>

#include <array>

namespace mini {
namespace integrator {

/**
 * @brief Gauss--Legendre quadrature rules, i.e. \f$ \int_{-1}^{1} f(\xi) \,\mathrm{d}\xi \approx \sum_{q=1}^{Q} w_q f(\xi_q) \f$
 * 
 * @tparam T  Type of scalar variables.
 * @tparam N  Nnumber of quadrature points, which must be chosen from { 1, 2, 3, 4, 5, 6 }.
 */
template <std::floating_point T, int N>
struct Legendre {
  using Scalar = T;
  using Array = std::array<T, N>;
  static const Array points;
  static const Array weights;
  static constexpr int Q = N;

  static_assert(1 <= Q && Q <= 6);

  static constexpr Array BuildPoints() {
    return _BuildPoints();
  }
  static constexpr Array BuildWeights() {
    return _BuildWeights();
  }

 private:
  static constexpr Array _BuildPoints() requires(Q == 1) {
    return { 0.0 };
  }
  static constexpr Array _BuildWeights() requires(Q == 1) {
    return { 2.0 };
  }

  static constexpr Array _BuildPoints() requires(Q == 2) {
    return { -std::sqrt(Scalar(1.0/3.0)), +std::sqrt(Scalar(1.0/3.0)) };
  }
  static constexpr Array _BuildWeights() requires(Q == 2) {
    return { 1.0, 1.0 };
  }

  static constexpr Array _BuildPoints() requires(Q == 3) {
    return { -std::sqrt(Scalar(0.6)), 0.0, +std::sqrt(Scalar(0.6)) };
  }
  static constexpr Array _BuildWeights() requires(Q == 3) {
    return { 5.0/9.0, 8.0/9.0, 5.0/9.0 };
  }

  static constexpr Array _BuildPoints() requires(Q == 4) {
    return {
        -std::sqrt((3 + 2 * std::sqrt(Scalar(1.2))) / 7),
        -std::sqrt((3 - 2 * std::sqrt(Scalar(1.2))) / 7),
        +std::sqrt((3 - 2 * std::sqrt(Scalar(1.2))) / 7),
        +std::sqrt((3 + 2 * std::sqrt(Scalar(1.2))) / 7),
    };
  }
  static constexpr Array _BuildWeights() requires(Q == 4) {
    return {
        (18 - std::sqrt(Scalar(30.0))) / 36,
        (18 + std::sqrt(Scalar(30.0))) / 36,
        (18 + std::sqrt(Scalar(30.0))) / 36,
        (18 - std::sqrt(Scalar(30.0))) / 36,
    };
  }

  static constexpr Array _BuildPoints() requires(Q == 5) {
    return {
        -std::sqrt((5 + std::sqrt(Scalar(40 / 7.0))) / 9),
        -std::sqrt((5 - std::sqrt(Scalar(40 / 7.0))) / 9),
        0,
        +std::sqrt((5 - std::sqrt(Scalar(40 / 7.0))) / 9),
        +std::sqrt((5 + std::sqrt(Scalar(40 / 7.0))) / 9),
    };
  }
  static constexpr Array _BuildWeights() requires(Q == 5) {
    return {
        (322 - 13 * std::sqrt(Scalar(70.0))) / 900,
        (322 + 13 * std::sqrt(Scalar(70.0))) / 900,
        128.0 / 225.0,
        (322 + 13 * std::sqrt(Scalar(70.0))) / 900,
        (322 - 13 * std::sqrt(Scalar(70.0))) / 900,
    };
  }

  static constexpr Array _BuildPoints() requires(Q == 6) {
    return {
        -0.932469514203152,
        -0.6612093864662645,
        -0.23861918608319693,
        +0.23861918608319693,
        +0.6612093864662645,
        +0.932469514203152,
    };
  }
  static constexpr Array _BuildWeights() requires(Q == 6) {
    return {
        0.17132449237917016,
        0.36076157304813855,
        0.4679139345726912,
        0.4679139345726912,
        0.36076157304813855,
        0.17132449237917016,
    };
  }
};
template <std::floating_point S, int Q>
typename Legendre<S, Q>::Array const
Legendre<S, Q>::points = Legendre<S, Q>::BuildPoints();

template <std::floating_point S, int Q>
typename Legendre<S, Q>::Array const
Legendre<S, Q>::weights = Legendre<S, Q>::BuildWeights();

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_LEGENDRE_HPP_
