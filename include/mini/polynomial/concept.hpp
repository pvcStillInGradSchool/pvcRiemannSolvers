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
  typename P::Integrator;
};

template <typename P, typename G>
concept HasGeneralMethods = requires(P const &cp, P *p, G const &g,
    typename P::Value (*f)(G const &)) {
  requires HasGeneralTypes<P>;

  requires std::same_as<G, typename P::Global>;
  { cp.GlobalToValue(g) } -> std::same_as<typename P::Value>;

  { cp.basis() } -> std::same_as<typename P::Basis const &>;
  { cp.coeff() } -> std::same_as<typename P::Coeff const &>;
  { p->SetZero() } -> std::same_as<void>;

  { cp.integrator() } -> std::same_as<typename P::Integrator const &>;

  { p->Approximate(f) } -> std::same_as<void>;
};

template <typename P>
concept General =
    HasGeneralTypes<P> &&
    HasGeneralMethods<P, typename P::Global>;

template <typename P>
concept Modal = requires(P const &cp, P *p) {
  requires General<P>;

  { cp.projection() };

  { cp.average() } -> std::same_as<typename P::Value>;
};

template <typename P>
concept Nodal = requires(P const &cp, P *p) {
  requires General<P>;

  { cp.interpolation() };

  { cp.GetValue(0) } -> std::same_as<typename P::Value>;
  { cp.LocalToValue(typename P::Local()) } -> std::same_as<typename P::Value>;

  { p->SetValue(0, typename P::Value()) } -> std::same_as<void>;
};

}  // namespace polynomial
}  // namespace mini

#endif  //  MINI_POLYNOMIAL_CONCEPT_HPP_
