//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_INTEGRATOR_FUNCTION_HPP_
#define MINI_INTEGRATOR_FUNCTION_HPP_

#include <concepts>

#include <cmath>

#include <type_traits>
#include <utility>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace integrator {

/**
 * @brief Perform Gaussian quadrature of a callable object on a Gauss object in the parametric space.
 * 
 * @tparam Callable the type of the integrand
 * @tparam Gauss the type of the gauss
 * @param local_to_value the integrand using local coordinates as arguments
 * @param gauss the Gauss object
 * @return auto the value of the integral
 */
template <typename Callable, typename Gauss>
auto Quadrature(Callable &&local_to_value, const Gauss &gauss) {
  using Local = typename Gauss::Local;
  static_assert(std::regular_invocable<Callable, Local>);
  using Value = std::invoke_result_t<Callable, Local>;
  Value sum; algebra::SetZero(&sum);
  auto n = gauss.CountPoints();
  for (int i = 0; i < n; ++i) {
    auto f_val = local_to_value(gauss.GetLocal(i));
    f_val *= gauss.GetLocalWeight(i);
    sum += f_val;
  }
  return sum;
}

/**
 * @brief Perform Gaussian quadrature of a callable object on a Gauss object in the physical space.
 * 
 * @tparam Callable the type of the integrand
 * @tparam Gauss the type of the gauss
 * @param global_to_value the integrand using global coordinates as arguments
 * @param gauss the Gauss object
 * @return auto the value of the integral
 */
template <typename Callable, typename Gauss>
auto Integrate(Callable &&global_to_value, const Gauss &gauss) {
  using Global = typename Gauss::Global;
  static_assert(std::regular_invocable<Callable, Global>);
  using Value = std::invoke_result_t<Callable, Global>;
  Value sum; algebra::SetZero(&sum);
  auto n = gauss.CountPoints();
  for (int i = 0; i < n; ++i) {
    auto f_val = global_to_value(gauss.GetGlobal(i));
    f_val *= gauss.GetGlobalWeight(i);
    sum += f_val;
  }
  return sum;
}

/**
 * @brief Calculate the inner-product of two functions on a Gauss object.
 * 
 * @tparam Func1 the type of the first function
 * @tparam Func2 the type of the second function
 * @tparam Gauss the type of the gauss
 * @param f1 the first function
 * @param f2 the second function
 * @param gauss the Gauss object
 * @return auto the value of the innerproduct
 */
template <typename Func1, typename Func2, typename Gauss>
auto Innerprod(Func1 &&f1, Func2 &&f2, const Gauss &gauss) {
  using Global = typename Gauss::Global;
  static_assert(std::regular_invocable<Func1, Global>);
  static_assert(std::regular_invocable<Func2, Global>);
  return Integrate([&f1, &f2](const Global &xyz_global){
    return f1(xyz_global) * f2(xyz_global);
  }, gauss);
}

/**
 * @brief Calculate the 2-norm of a function on a Gauss object.
 * 
 * @tparam Callable type of the function
 * @tparam Gauss the type of the gauss
 * @param f the function
 * @param gauss the Gauss object
 * @return auto the value of the norm
 */
template <typename Callable, typename Gauss>
auto Norm(Callable &&f, const Gauss &gauss) {
  auto ip = Innerprod(std::forward<Callable>(f), std::forward<Callable>(f),
      gauss);
  return std::sqrt(ip);
}

/**
 * @brief Change a group of linearly independent functions into an orthonormal basis.
 * 
 * @tparam Basis the type of the basis
 * @tparam Gauss the type of the gauss
 * @param basis the basis to be orthonormalized, whose components are linearly independent from each other
 * @param gauss the Gauss object
 */
template <class Basis, class Gauss>
void OrthoNormalize(Basis *basis, const Gauss &gauss) {
  constexpr int N = Basis::N;
  using MatNxN = typename Basis::MatNxN;
  using Global = typename Gauss::Global;
  using Scalar = typename Gauss::Real;
  MatNxN S; S.setIdentity();
  auto A = Integrate([basis](const Global &xyz){
    auto col = (*basis)(xyz);
    MatNxN result = col * col.transpose();
    return result;
  }, gauss);
  S(0, 0) = 1 / std::sqrt(A(0, 0));
  for (int i = 1; i < N; ++i) {
    for (int j = 0; j < i; ++j) {
      Scalar temp = 0;
      for (int k = 0; k <= j; ++k) {
        temp += S(j, k) * A(k, i);
      }
      for (int l = 0; l <= j; ++l) {
        S(i, l) -= temp * S(j, l);
      }
    }
    Scalar norm_sq = 0;
    for (int j = 0; j <= i; ++j) {
      Scalar sum = 0, Sij = S(i, j);
      for (int k = 0; k < j; ++k) {
        sum += 2 * S(i, k) * A(k, j);
      }
      norm_sq += Sij * (Sij * A(j, j) + sum);
    }
    S.row(i) /= std::sqrt(norm_sq);
  }
  basis->Transform(algebra::GetLowerTriangularView(S));
}

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_FUNCTION_HPP_
