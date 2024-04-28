//  Copyright 2023 PEI Weicheng
#ifndef MINI_GAUSS_JACOBI_HPP_
#define MINI_GAUSS_JACOBI_HPP_

#include <concepts>

#include <array>

namespace mini {
namespace integrator {

/**
 * @brief Gauss--Jacobi quadrature rules, i.e. \f$ \int_{-1}^{1} f(\xi) (1-\xi)^\alpha (1+\xi)^\beta \,\mathrm{d}\xi \approx \sum_{q=1}^{Q} w_q f(\xi_q) \f$
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam Q  Nnumber of quadrature points, which must be chosen from { 1, 2, 3, 4, 5 }.
 * @tparam kAlpha  Power of \f$ (1-\xi) \f$ in the integrand.
 * @tparam kBeta  Power of \f$ (1+\xi) \f$ in the integrand.
 */
template <std::floating_point Scalar, int Q, int kAlpha, int kBeta>
struct Jacobi {
  using Array = std::array<Scalar, Q>;
  static const Array points;
  static const Array weights;

  static_assert(1 <= Q && Q <= 5);

  static constexpr Array BuildPoints() {
    return _BuildPoints();
  }
  static constexpr Array BuildWeights() {
    return _BuildWeights();
  }

 private:
  static constexpr Array _BuildPoints() requires(Q == 1 && kAlpha == 2 && kBeta == 0) {
    return { -0.5 };
  }
  static constexpr Array _BuildWeights() requires(Q == 1 && kAlpha == 2 && kBeta == 0) {
    return { 2.6666666666666665 };
  }

  static constexpr Array _BuildPoints() requires(Q == 2 && kAlpha == 2 && kBeta == 0) {
    return {
      -0.754970354689117,
      0.08830368802245062
    };
  }
  static constexpr Array _BuildWeights() {
    return {
      1.860379610028064,
      0.8062870566386026
    };
  }

  static constexpr Array _BuildPoints() requires(Q == 3 && kAlpha == 2 && kBeta == 0) {
    return {
      -0.8540119518537006,
      -0.30599246792329643,
      0.41000441977699675
    };
  }
  static constexpr Array _BuildWeights() requires(Q == 3 && kAlpha == 2 && kBeta == 0) {
    return {
      1.2570908885190917,
      1.169970154078929,
      0.23960562406864572
    };
  }

  static constexpr Array _BuildPoints() requires(Q == 4 && kAlpha == 2 && kBeta == 0) {
    return {
      -0.9029989011060054,
      -0.5227985248962753,
      0.03409459020873491,
      0.5917028357935458
    };
  }
  static constexpr Array _BuildWeights() requires(Q == 4 && kAlpha == 2 && kBeta == 0) {
    return {
      0.8871073248902219,
      1.1476703183937156,
      0.5490710973833848,
      0.08281792599934465
    };
  }

  static constexpr Array _BuildPoints() requires(Q == 5 && kAlpha == 2 && kBeta == 0) {
    return {
      -0.9308421201635698,
      -0.6530393584566087,
      -0.2202272258689614,
      0.26866694526177365,
      0.7021084258940329
    };
  }
  static constexpr Array _BuildWeights() requires(Q == 5 && kAlpha == 2 && kBeta == 0) {
    return {
      0.6541182742861681,
      1.009591695199291,
      0.7136012897727205,
      0.25644480578369516,
      0.03291060162479203,
    };
  }
};
template <std::floating_point Scalar, int Q, int A, int B>
typename Jacobi<Scalar, Q, A, B>::Array const
Jacobi<Scalar, Q, A, B>::points = Jacobi<Scalar, Q, A, B>::BuildPoints();

template <std::floating_point Scalar, int Q, int A, int B>
typename Jacobi<Scalar, Q, A, B>::Array const
Jacobi<Scalar, Q, A, B>::weights = Jacobi<Scalar, Q, A, B>::BuildWeights();

}  // namespace integrator
}  // namespace mini

#endif  // MINI_GAUSS_JACOBI_HPP_
