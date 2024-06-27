//  Copyright 2024 PEI Weicheng
#ifndef MINI_POLYNOMIAL_EXPANSION_HPP_
#define MINI_POLYNOMIAL_EXPANSION_HPP_

#include <concepts>

#include <cmath>
#include <cstring>

#include <iostream>
#include <type_traits>
#include <utility>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace polynomial {

/**
 * @brief A vector-valued function projected onto a given `Basis` defined in `mini::basis`.
 * 
 * @tparam kComponents the number of function components
 * @tparam Basis the type of the basis
 */
template <int kComponents, class Basis>
class Expansion {
 public:
  using Scalar = typename Basis::Scalar;
  static constexpr int N = Basis::N;
  static constexpr int K = kComponents;
  static constexpr int kFields = K * N;

  template<int kRows, int kColumns>
  using Matrix = algebra::Matrix<Scalar, kRows, kColumns>;

  using Coeff = Matrix<K, N>;
  using Value = Matrix<K, 1>;

 protected:
  Coeff coeff_;

 public:
  Expansion() = default;
  Expansion(const Expansion &) = default;
  Expansion &operator=(const Expansion &) = default;
  Expansion &operator=(Expansion &&that) noexcept = default;
  Expansion(Expansion &&that) noexcept = default;
  ~Expansion() noexcept = default;

  Coeff const &coeff() const {
    return coeff_;
  }

  void SetCoeff(Coeff const &coeff) {
    coeff_ = coeff;
  }

  template <typename FieldIndexToScalar>
  void SetCoeff(FieldIndexToScalar && field_index_to_scalar) {
    for (int i_field = 0; i_field < kFields; ++i_field) {
      coeff_.reshaped()[i_field] = field_index_to_scalar(i_field);
    }
  }

  void SetZero() {
    coeff_.setZero();
  }

  Scalar GetScalar(int i_field) const {
    return coeff_.reshaped()[i_field];
  }

  const Scalar *GetCoeffFrom(const Scalar *input) {
    std::memcpy(coeff_.data(), input, sizeof(Scalar) * coeff_.size());
    return input + coeff_.size();
  }
  Scalar *WriteCoeffTo(Scalar *output) const {
    std::memcpy(output, coeff_.data(), sizeof(Scalar) * coeff_.size());
    return output + coeff_.size();
  }
  static void AddCoeffTo(Coeff const &coeff, Scalar *output) {
    for (int c = 0; c < N; ++c) {
      for (int r = 0; r < K; ++r) {
        *output++ += coeff(r, c);
      }
    }
  }
  static void MinusCoeff(Coeff const &coeff, Scalar *output) {
    for (int c = 0; c < N; ++c) {
      for (int r = 0; r < K; ++r) {
        *output++ -= coeff(r, c);
      }
    }
  }

  Expansion &operator+=(Coeff const &coeff) {
    coeff_ += coeff;
    return *this;
  }
  Expansion &operator*=(Scalar ratio) {
    coeff_ *= ratio;
    return *this;
  }
  Expansion &operator/=(Scalar ratio) {
    coeff_ /= ratio;
    return *this;
  }
  Expansion &operator*=(const Value& ratio) {
    for (int k = 0; k < K; ++k) {
      coeff_.row(k) *= ratio[k];
    }
    return *this;
  }
  Expansion &LeftMultiply(Matrix<K, K> const &left) {
    Coeff temp = left * coeff_;
    coeff_ = temp;
    return *this;
  }
};

}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_EXPANSION_HPP_
