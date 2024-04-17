//  Copyright 2023 PEI Weicheng
#ifndef MINI_GAUSS_LOBATTO_HPP_
#define MINI_GAUSS_LOBATTO_HPP_

#include <concepts>

#include <cmath>

#include <array>

namespace mini {
namespace gauss {

/**
 * @brief Gauss--Lobatto quadrature rules, i.e. \f$ \int_{-1}^{1} f(\xi) \,\mathrm{d}\xi \approx \sum_{q=1}^{Q} w_q f(\xi_q) \f$, in which \f$ \xi_1 = -1 \f$ and \f$ \xi_Q = +1 \f$.
 * 
 * @tparam T  Type of scalar variables.
 * @tparam N  Nnumber of quadrature points, which must be chosen from { 2, 3, 4, 5, 6 }.
 */
template <std::floating_point T, int N>
struct Lobatto {
  using Scalar = T;
  using Array = std::array<T, N>;
  static const Array points;
  static const Array weights;
  static constexpr int Q = N;

  static_assert(2 <= Q && Q <= 6);

  static constexpr Array BuildPoints() {
    return _BuildPoints();
  }
  static constexpr Array BuildWeights() {
    return _BuildWeights();
  }

 private:
  static constexpr Array _BuildPoints() requires(Q == 2) {
    return { -1.0, +1.0 };
  }
  static constexpr Array _BuildWeights() requires(Q == 2) {
    return { 1.0, 1.0 };
  }

  static constexpr Array _BuildPoints() requires(Q == 3) {
    return { -1.0, 0.0, +1.0 };
  }
  static constexpr Array _BuildWeights() requires(Q == 3) {
    return { 1.0/3, 4.0/3, 1.0/3 };
  }

  static constexpr Array _BuildPoints() requires(Q == 4) {
    return { -1.0, -std::sqrt(1.0/5), +std::sqrt(1.0/5), +1.0 };
  }
  static constexpr Array _BuildWeights() requires(Q == 4) {
    return { 1.0/6, 5.0/6, 5.0/6, 1.0/6 };
  }

  static constexpr Array _BuildPoints() requires(Q == 5) {
    return { -1.0, -std::sqrt(21.0)/7, 0.0, +std::sqrt(21.0)/7, +1.0 };
  }
  static constexpr Array _BuildWeights() requires(Q == 5) {
    return { 0.1, 49.0/90, 32.0/45, 49.0/90, 0.1 };
  }

  static constexpr Array _BuildPoints() requires(Q == 6) {
    return {
        -1.0,
        -std::sqrt((7 + 2 * std::sqrt(7.0)) / 21),
        -std::sqrt((7 - 2 * std::sqrt(7.0)) / 21),
        +std::sqrt((7 - 2 * std::sqrt(7.0)) / 21),
        +std::sqrt((7 + 2 * std::sqrt(7.0)) / 21),
        +1.0
    };
  }
  static constexpr Array _BuildWeights() requires(Q == 6) {
    return {
        1.0 / 15,
        (14.0 - std::sqrt(7.0)) / 30,
        (14.0 + std::sqrt(7.0)) / 30,
        (14.0 + std::sqrt(7.0)) / 30,
        (14.0 - std::sqrt(7.0)) / 30,
        1.0 / 15
    };
  }
};
template <std::floating_point S, int Q>
typename Lobatto<S, Q>::Array const
Lobatto<S, Q>::points = Lobatto<S, Q>::BuildPoints();

template <std::floating_point S, int Q>
typename Lobatto<S, Q>::Array const
Lobatto<S, Q>::weights = Lobatto<S, Q>::BuildWeights();

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_LOBATTO_HPP_
