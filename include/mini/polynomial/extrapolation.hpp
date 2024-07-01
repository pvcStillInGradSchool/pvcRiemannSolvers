//  Copyright 2024 PEI Weicheng
#ifndef MINI_POLYNOMIAL_EXTRAPOLATION_HPP_
#define MINI_POLYNOMIAL_EXTRAPOLATION_HPP_

#include <concepts>

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
  using Local = typename Interpolation::Local;
  using Value = typename Interpolation::Value;
  using Integrator = typename Interpolation::Integrator;

  static constexpr int D = Interpolation::D;
  static constexpr int P = Interpolation::P;
  static constexpr int K = Interpolation::K;
  static constexpr int N = Interpolation::N;

  using Projection = polynomial::Projection<Scalar, D, P, K>;
  static constexpr int M = Projection::N;  // number of modal basis

  using Wrapper = typename Projection::Wrapper;

 private:
  using MatKxN = typename Interpolation::Coeff;
  using MatKxM = typename Projection::Coeff;
  using MatNxM = algebra::Matrix<Scalar, N, M>;
  using MatNx1 = algebra::Matrix<Scalar, N, 1>;
  using Mat1xM = algebra::Matrix<Scalar, 1, M>;
  static constexpr int kFields = K * N;

  Projection projection_;

  /**
   * @brief The matrix for conversions between the two representations.
   * 
   * The fundamental identity is
   *    value = nodal_coeff_row @ nodal_basis_col
   *          = modal_coeff_row @ modal_basis_col
   * The matrix acts as
   *    nodal_basis_col = modal_to_nodal_ * modal_basis_col
   * So, there is
   *    nodal_coeff_row * modal_to_nodal_ = modal_coeff_row
   */
  MatNxM modal_to_nodal_;

  /**
   * @brief Keep the two representations consistent.
   * 
   * TODO(PVC): check consistency before invoking the expensive matrix production.
   */
  void UpdateModalCoeff() {
    projection_.SetCoeff(this->GetValues() * modal_to_nodal_);
  }
  void UpdateNodalCoeff() {
    for (int i = 0; i < N; ++i) {
      auto const &global_i = this->integrator().GetGlobal(i);
      this->Interpolation::SetValue(i, projection_.GlobalToValue(global_i));
    }
  }

 public:
  explicit Extrapolation(typename Integrator::Base const &integrator)
      : Interpolation(integrator), projection_(integrator) {
    auto integrand = [this](Local const &local) -> MatNxM {
      // TODO(PVC): use Kronecker delta property
      MatNx1 nodal_basis_values = this->Interpolation::basis().GetValues(local);
      Global global = this->coordinate().LocalToGlobal(local);
      Mat1xM modal_basis_values = this->projection_.basis()(global);
      MatNxM product = nodal_basis_values * modal_basis_values;
      product *= this->coordinate().LocalToJacobian(local).determinant();
      return product;
    };
    modal_to_nodal_ = mini::integrator::Quadrature(integrand, integrator);
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

  template <typename Callable>
  void Approximate(Callable &&global_to_value) {
    Interpolation::Approximate(std::forward<Callable>(global_to_value));
    UpdateModalCoeff();
  }

  const Scalar *GetCoeffFrom(const Scalar *input) {
    auto output = Interpolation::GetCoeffFrom(input);
    UpdateModalCoeff();
    return output;
  }

  Interpolation const &interpolation() const {
    return *this;
  }

  Projection const &projection() const {
    return projection_;
  }

  void SetCoeff(typename Projection::Coeff const &coeff) {
    projection_.SetCoeff(coeff);
    UpdateNodalCoeff();
  }

  void SetCoeff(typename Interpolation::Coeff const &coeff) {
    this->Interpolation::SetCoeff(coeff);
    UpdateModalCoeff();
  }

  template <typename FieldIndexToScalar>
      requires std::same_as< Scalar,
          std::invoke_result_t<FieldIndexToScalar, int> >
  void SetCoeff(FieldIndexToScalar &&field_index_to_scalar) {
    this->Interpolation::SetCoeff(
        std::forward<FieldIndexToScalar>(field_index_to_scalar));
    UpdateModalCoeff();
  }

  /**
   * @brief Almost the same as `Interpolation::SetCoeff`, except calling `UpdateModalCoeff` when `i_node + 1 == N`.
   * 
   */
  void SetCoeff(int i_basis, Value const &coeff_i) {
    this->Interpolation::SetCoeff(i_basis, coeff_i);
    if (i_basis + 1 == N) {
      UpdateModalCoeff();
    }
  }

  /**
   * @brief Almost the same as `Interpolation::SetValue`, except calling `UpdateModalCoeff` when `i_node + 1 == N`.
   * 
   */
  void SetValue(int i_basis, Value const &value_i) {
    this->Interpolation::SetValue(i_basis, value_i);
    if (i_basis + 1 == N) {
      UpdateModalCoeff();
    }
  }

  void SetZero() {
    this->Interpolation::SetZero();
    projection_.SetZero();
  }

  Value average() const {
    return projection_.average();
  }

  /**
   * @brief A wrapper of `projection().GlobalToValue()`.
   * 
   */
  Value GlobalToValue(Global const &global) const {
    return projection_.GlobalToValue(global);
  }

  void LocalToGlobalAndValue(Local const &local,
      Global *global, Value *value) const {
    projection_.LocalToGlobalAndValue(local, global, value);
  }

};

}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_HEXAHEDRON_HPP_
