//  Copyright 2023 PEI Weicheng

#ifndef MINI_RIEMANN_CONCEPT_HPP_
#define MINI_RIEMANN_CONCEPT_HPP_

#include <concepts>
#include <type_traits>

namespace mini {
namespace riemann {

template <typename R>
concept HasConvectiveData = requires {
  requires std::integral<decltype(R::kComponents)>;
  requires std::integral<decltype(R::kDimensions)>;

  typename R::Scalar;
  requires std::floating_point<typename R::Scalar>;

  typename R::Vector;
  typename R::Conservative;
  typename R::Flux;
  typename R::FluxMatrix;
};

template <typename R, typename C>
concept HasConvectiveMethods = requires(R riemann, C const &value) {
  requires HasConvectiveData<R>;
  requires std::same_as<C, typename R::Conservative>;
  { R::GetFluxMatrix(value) } -> std::same_as<typename R::FluxMatrix>;
  { riemann.GetFluxUpwind(value, value) } -> std::same_as<typename R::Flux>;
};

template <typename R>
concept Convective =
    HasConvectiveData<R> &&
    HasConvectiveMethods<R, typename R::Conservative>;

template <typename R>
concept HasDiffusiveData = requires {
  requires HasConvectiveData<R>;
  // more than convective
  typename R::Gradient;
};

template <typename R, typename C, typename G, typename F, typename M,
    typename S, typename V>
concept HasDiffusiveMethods = requires(R riemann, C const &value,
    G const &gradient, F *flux, M *flux_matrix, S distance, V const &normal) {
  requires std::same_as<G, typename R::Gradient>;
  requires std::same_as<S, typename R::Scalar>;
  requires std::same_as<V, typename R::Vector>;
  requires std::same_as<F, typename R::Flux>;
  requires std::same_as<M, typename R::FluxMatrix>;
  { R::MinusViscousFlux(value, gradient, flux_matrix) } -> std::same_as<void>;
  { R::MinusViscousFlux(value, gradient, normal, flux) } -> std::same_as<void>;
  { R::GetCommonGradient(distance, normal, value, value,
      gradient, gradient) } -> std::same_as<typename R::Gradient>;
};

template <typename R>
concept Diffusive =
    HasDiffusiveData<R> &&
    HasDiffusiveMethods<R, typename R::Conservative, typename R::Gradient,
        typename R::Flux, typename R::FluxMatrix,
        typename R::Scalar, typename R::Vector>;

template <typename R>
concept ConvectiveDiffusive = Convective<R> && Diffusive<R>;

template <Convective C, Diffusive D>
class ConvectionDiffusion : public C, public D {
 public:
  static constexpr int kDimensions = C::kDimensions;
  static_assert(kDimensions == D::kDimensions);

  static constexpr int kComponents = C::kComponents;
  static_assert(kComponents == D::kComponents);

  using Scalar = typename C::Scalar;
  static_assert(std::is_same_v<Scalar, typename D::Scalar>);

  using Vector = typename C::Vector;
  static_assert(std::is_same_v<Vector, typename D::Vector>);

  using Conservative = typename C::Conservative;
  static_assert(std::is_same_v<Conservative, typename D::Conservative>);

  using FluxMatrix = typename C::FluxMatrix;
  static_assert(std::is_same_v<FluxMatrix, typename D::FluxMatrix>);

  using Flux = typename C::Flux;
  static_assert(std::is_same_v<Flux, typename D::Flux>);

  using Gradient = typename D::Gradient;
};

}  // namespace riemann
}  // namespace mini

#endif  //  MINI_RIEMANN_CONCEPT_HPP_
