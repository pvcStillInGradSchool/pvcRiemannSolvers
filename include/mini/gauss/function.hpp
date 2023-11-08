//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_GAUSS_FUNCTION_HPP_
#define MINI_GAUSS_FUNCTION_HPP_

#include <concepts>

#include <cmath>

#include <type_traits>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace gauss {

/**
 * @brief Perform Gaussian quadrature of a callable object on an integratable element in the parametric space.
 * 
 * @tparam Callable the type of the integrand
 * @tparam Element the type of the gauss
 * @param f_in_local the integrand using local coordinates as arguments
 * @param element the gauss
 * @return auto the value of the integral
 */
template <typename Callable, typename Element>
auto Quadrature(Callable &&f_in_local, Element &&element) {
  using E = std::remove_reference_t<Element>;
  using Local = typename E::Local;
  decltype(f_in_local(Local())) sum; algebra::SetZero(&sum);
  auto n = element.CountPoints();
  for (int i = 0; i < n; ++i) {
    auto f_val = f_in_local(element.GetLocalCoord(i));
    f_val *= element.GetLocalWeight(i);
    sum += f_val;
  }
  return sum;
}

/**
 * @brief Perform Gaussian quadrature of a callable object on an integratable element in the physical space.
 * 
 * @tparam Callable the type of the integrand
 * @tparam Element the type of the gauss
 * @param f_in_global the integrand using global coordinates as arguments
 * @param element the gauss
 * @return auto the value of the integral
 */
template <typename Callable, typename Element>
auto Integrate(Callable &&f_in_global, Element &&element) {
  using E = std::remove_reference_t<Element>;
  using Global = typename E::Global;
  decltype(f_in_global(Global())) sum; algebra::SetZero(&sum);
  auto n = element.CountPoints();
  auto const &gauss = element;
  for (int i = 0; i < n; ++i) {
    auto f_val = f_in_global(gauss.GetGlobalCoord(i));
    f_val *= gauss.GetGlobalWeight(i);
    sum += f_val;
  }
  return sum;
}

/**
 * @brief Calculate the inner-product of two functions on an integratable element.
 * 
 * @tparam Func1 the type of the first function
 * @tparam Func2 the type of the second function
 * @tparam Element the type of the gauss
 * @param f1 the first function
 * @param f2 the second function
 * @param element the gauss
 * @return auto the value of the innerproduct
 */
template <typename Func1, typename Func2, typename Element>
auto Innerprod(Func1 &&f1, Func2 &&f2, Element &&element) {
  using E = std::remove_reference_t<Element>;
  using Global = typename E::Global;
  return Integrate([&f1, &f2](const Global &xyz_global){
    return f1(xyz_global) * f2(xyz_global);
  }, element);
}

/**
 * @brief Calculate the 2-norm of a function on an integratable element.
 * 
 * @tparam Callable type of the function
 * @tparam Element the type of the gauss
 * @param f the function
 * @param element the gauss
 * @return auto the value of the norm
 */
template <typename Callable, typename Element>
auto Norm(Callable &&f, Element &&element) {
  return std::sqrt(Innerprod(f, f, element));
}

/**
 * @brief Change a group of linearly independent functions into an orthonormal basis.
 * 
 * @tparam Basis the type of the basis
 * @tparam Element the type of the gauss
 * @param basis the basis to be orthonormalized, whose components are linearly independent from each other
 * @param elem the gauss
 */
template <class Basis, class Element>
void OrthoNormalize(Basis *basis, const Element &elem) {
  constexpr int N = Basis::N;
  using MatNxN = typename Basis::MatNxN;
  using MatDx1 = typename Element::Global;
  using Scalar = typename Element::Real;
  MatNxN S; S.setIdentity();
  auto A = Integrate([basis](const MatDx1 &xyz){
    auto col = (*basis)(xyz);
    MatNxN result = col * col.transpose();
    return result;
  }, elem);
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

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_FUNCTION_HPP_
