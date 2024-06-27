//  Copyright 2023 PEI Weicheng
#ifndef MINI_BASIS_VINCENT_HPP_
#define MINI_BASIS_VINCENT_HPP_

#include <concepts>

#include <cassert>
#include <cmath>

namespace mini {
namespace basis {

/**
 * @brief The \f$ g_\mathrm{left}(\xi) \f$ and \f$ g_\mathrm{right}(\xi) \f$ in Vincent's ESFR schemes.
 * 
 * @tparam S The type of scalar variables.
 */
template <std::floating_point S>
class Vincent {
 public:
  using Scalar = S;

 private:
  Scalar c_prev_, c_next_;
  int degree_;

 public:
  static Scalar DiscontinuousGalerkin(int degree) {
    return 1.0;
  }
  static Scalar HuynhLumpingLobatto(int degree) {
    return degree / (2.0 * degree + 1);
  }

  /**
   * @brief Construct a new Vincent object
   * 
   * @param degree the degree of the polynomial to be corrected
   * @param c_next the ratio of \f$ \mathrm{P}_{k+1}(\xi) \f$
   */
  Vincent(int degree, Scalar c_next)
      : c_prev_(1 - c_next), c_next_(c_next), degree_(degree) {
    assert(degree > 0);
  }

  /**
   * @brief Get the value of \f$ g_\mathrm{right}(\xi) = \frac12 \left( \mathrm{P}_{k}(\xi) + c_\mathrm{prev} \mathrm{P}_{k-1}(\xi) + c_\mathrm{next} \mathrm{P}_{k+1}(\xi) \right) \f$.
   * 
   * @param local the value of \f$ \xi \f$
   * @return Scalar the value of \f$ g_\mathrm{right}(\xi) \f$
   */
  Scalar LocalToRightValue(Scalar local) const {
    Scalar value = std::legendre(degree_, local);
    value += c_prev_ * std::legendre(degree_ - 1, local);
    value += c_next_ * std::legendre(degree_ + 1, local);
    value *= 0.5;
    return value;
  }
  /**
   * @brief Get the value of \f$ g_\mathrm{left}(\xi) = g_\mathrm{right}(-\xi) \f$.
   * 
   * @param local the value of \f$ \xi \f$
   * @return Scalar the value of \f$ g_\mathrm{left}(\xi) \f$
   */
  Scalar LocalToLeftValue(Scalar local) const {
    return LocalToRightValue(-local);
  }

  /**
   * @brief Get the value of \f$ \frac{\mathrm{d}}{\mathrm{d}\xi} g_\mathrm{right}(\xi) \f$.
   * 
   * @param local the value of \f$ \xi \f$
   * @return Scalar the value of \f$ \frac{\mathrm{d}}{\mathrm{d}\xi} g_\mathrm{right}(\xi) \f$
   */
  Scalar LocalToRightDerivative(Scalar local) const {
    Scalar legendre_derivative_prev = 0.0;
    Scalar legendre_derivative_curr = 0.0;
    Scalar legendre_derivative_next = 0.0;
    for (int k_curr = 0; k_curr <= degree_; ++k_curr) {
      if (k_curr > 0) {
        legendre_derivative_prev = legendre_derivative_curr;
        legendre_derivative_curr = legendre_derivative_next;
      }
      int k_next = k_curr + 1;
      legendre_derivative_next = (k_next * std::legendre(k_curr, local)
          + local * legendre_derivative_curr);
    }
    return 0.5 * (legendre_derivative_curr +
        c_prev_ * legendre_derivative_prev +
        c_next_ * legendre_derivative_next);
  }

  /**
   * @brief Get the value of \f$ \frac{\mathrm{d}}{\mathrm{d}\xi} g_\mathrm{left}(\xi) \f$.
   * 
   * @param local the value of \f$ \xi \f$
   * @return Scalar the value of \f$ \frac{\mathrm{d}}{\mathrm{d}\xi} g_\mathrm{left}(\xi) \f$
   */
  Scalar LocalToLeftDerivative(Scalar local) const {
    return -LocalToRightDerivative(-local);
  }
};

}  // namespace basis
}  // namespace mini

#endif  // MINI_BASIS_VINCENT_HPP_
