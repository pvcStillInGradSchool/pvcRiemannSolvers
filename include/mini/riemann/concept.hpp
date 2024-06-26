//  Copyright 2023 PEI Weicheng

#ifndef MINI_RIEMANN_CONCEPT_HPP_
#define MINI_RIEMANN_CONCEPT_HPP_

#include <concepts>
#include <type_traits>

namespace mini {
namespace riemann {

template <typename R>
concept HasConvectiveData = requires {
  typename R::Convection;

  requires std::integral<decltype(R::kComponents)>;
  requires std::integral<decltype(R::kDimensions)>;

  typename R::Scalar;
  requires std::floating_point<typename R::Scalar>;

  typename R::Vector;
  typename R::Conservative;
  typename R::Flux;
  typename R::FluxMatrix;

  typename R::Jacobian;
};

template <typename R, typename C>
concept HasConvectiveMethods = requires(R riemann, C const &value) {
  requires HasConvectiveData<R>;
  requires std::same_as<C, typename R::Conservative>;
  { R::GetFluxMatrix(value) } -> std::same_as<typename R::FluxMatrix>;
  { riemann.GetFluxUpwind(value, value) }
      // e.g. riemann::euler::Flux is derived from algebra::eigen::Vector
      -> std::derived_from<typename R::Flux>;
};

template <typename R>
concept Convective =
    HasConvectiveData<R> &&
    HasConvectiveMethods<R, typename R::Conservative>;

template <typename R>
concept HasDiffusiveData = requires {
  typename R::Diffusion;

  requires std::integral<decltype(R::kComponents)>;
  requires std::integral<decltype(R::kDimensions)>;

  typename R::Scalar;
  requires std::floating_point<typename R::Scalar>;

  typename R::Vector;
  typename R::Conservative;
  typename R::Flux;
  typename R::FluxMatrix;

  typename R::Gradient;
  typename R::Property;
};

template <typename R, typename P, typename V, typename G, typename F,
    typename M, typename S, typename N>
concept HasDiffusiveMethods = requires(R riemann, P property, V const &value,
    G const &gradient, F *flux, M *flux_matrix, S distance, N const &normal) {
  requires std::same_as<P, typename R::Property>;
  requires std::same_as<G, typename R::Gradient>;
  requires std::same_as<S, typename R::Scalar>;
  requires std::same_as<N, typename R::Vector>;
  requires std::same_as<F, typename R::Flux>;
  requires std::same_as<M, typename R::FluxMatrix>;
  { R::MinusViscousFlux(flux_matrix, property, value, gradient) }
      -> std::same_as<void>;
  { R::MinusViscousFlux(flux, property, value, gradient, normal) }
      -> std::same_as<void>;
  { riemann.GetCommonGradient(normal, value, value, gradient, gradient) } -> std::same_as<G>;
};

template <typename R>
concept Diffusive =
    HasDiffusiveData<R> &&
    HasDiffusiveMethods<R, typename R::Property,
        typename R::Conservative, typename R::Gradient,
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

  using Convection = C;
  using Diffusion = D;

  using Scalar = typename C::Scalar;
  static_assert(std::is_same_v<Scalar, typename D::Scalar>);

  using Vector = typename C::Vector;
  static_assert(std::is_same_v<Vector, typename D::Vector>);

  using Conservative = typename D::Conservative;
  static_assert(std::derived_from<typename C::Conservative, Conservative>);

  using FluxMatrix = typename C::FluxMatrix;
  static_assert(std::is_same_v<FluxMatrix, typename D::FluxMatrix>);

  using Flux = typename D::Flux;
  static_assert(std::derived_from<typename C::Flux, Flux>);

  using Gradient = typename D::Gradient;
};

}  // namespace riemann
}  // namespace mini

#endif  //  MINI_RIEMANN_CONCEPT_HPP_
