//  Copyright 2024 PEI Weicheng
#ifndef MINI_POLYNOMIAL_EXTRAPOLATION_HPP_
#define MINI_POLYNOMIAL_EXTRAPOLATION_HPP_

#include "mini/polynomial/projection.hpp"

namespace mini {
namespace polynomial {

/**
 * @brief An adapter of `Interpolation` which can perform extrapolation like `Projection`.
 * 
 * @tparam Interpolation 
 */
template <class Interpolation>
class Extrapolation : public Interpolation {
 public:
  using Scalar = typename Interpolation::Scalar;
  using Global = typename Interpolation::Global;
  using Value = typename Interpolation::Value;
  using Integrator = typename Interpolation::IntegratorBase;

  static constexpr int D = Interpolation::D;
  static constexpr int P = Interpolation::P;
  static constexpr int K = Interpolation::K;
  static constexpr int N = Interpolation::N;

  using Projection = polynomial::Projection<Scalar, D, P, K>;
  static constexpr int M = Projection::N;  // number of modal basis

 private:
  using MatKxN = typename Interpolation::Coeff;
  using MatKxM = typename Projection::Coeff;
  using MatNxM = algebra::Matrix<Scalar, N, M>;

  Projection projection_;

 public:
  explicit Extrapolation(Integrator const &integrator)
      : Interpolation(integrator), projection_(integrator) {
  }

  Extrapolation() = default;
  Extrapolation(const Extrapolation &) = default;
  Extrapolation(Extrapolation &&) noexcept = default;
  Extrapolation &operator=(const Extrapolation &) = default;
  Extrapolation &operator=(Extrapolation &&) noexcept = default;
  ~Extrapolation() noexcept = default;

  Value Extrapolate(Global const &global) {
    return projection_.GlobalToValue(global);
  }
};

}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_HEXAHEDRON_HPP_
