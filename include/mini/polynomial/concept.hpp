//  Copyright 2024 PEI Weicheng
#ifndef MINI_POLYNOMIAL_CONCEPT_HPP_
#define MINI_POLYNOMIAL_CONCEPT_HPP_

#include <concepts>
#include <type_traits>

namespace mini {
namespace polynomial {

template <typename P>
concept HasGeneralTypes = requires {
  requires std::integral<decltype(P::D)>;
  requires std::integral<decltype(P::P)>;
  requires std::integral<decltype(P::K)>;
  requires std::integral<decltype(P::N)>;

  typename P::Scalar;
  requires std::floating_point<typename P::Scalar>;

  typename P::Local;
  typename P::Global;
  typename P::Value;
  typename P::Coeff;
  typename P::Basis;
};

template <typename P, typename G>
concept HasGeneralMethods = requires(P p, G const &g,
    typename P::Value (*f)(G const &)) {
  requires HasGeneralTypes<P>;

  requires std::same_as<G, typename P::Global>;
  { p.GlobalToValue(g) } -> std::same_as<typename P::Value>;

  { p.Approximate(f) } -> std::same_as<void>;
};

template <typename P>
concept General =
    HasGeneralTypes<P> &&
    HasGeneralMethods<P, typename P::Global>;

template <typename P>
concept Modal = requires(P p) {
  requires General<P>;

  { p.projection() };

  { p.average() } -> std::same_as<typename P::Value>;
};

}  // namespace polynomial
}  // namespace mini

#endif  //  MINI_POLYNOMIAL_CONCEPT_HPP_
