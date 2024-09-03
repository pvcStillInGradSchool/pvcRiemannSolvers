// Copyright 2023 PEI Weicheng
#ifndef MINI_TEMPORAL_ODE_HPP_
#define MINI_TEMPORAL_ODE_HPP_

#include <cassert>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace temporal {

/**
 * @brief The abstract base of all ODE systems.
 * 
 * @tparam Scalar the type of scalar variables.
 */
template<typename Scalar>
class System {
 public:
  using Column = algebra::DynamicVector<Scalar>;

  /**
   * @brief Set the current time value.
   * 
   * @param t_curr 
   */
  virtual void SetTime(double t_curr) = 0;

  /**
   * @brief Overwrite the solution by a given Column.
   * 
   */
  virtual void SetSolutionColumn(Column const &) = 0;

  /**
   * @brief Get a copy of the solution as a Column.
   * 
   * @return Column 
   */
  virtual Column GetSolutionColumn() const = 0;

  /**
   * @brief Get a copy of the residual as a Column.
   * 
   * @return Column 
   */
  virtual Column GetResidualColumn() const = 0;
};

/**
 * @brief The simplest ODE System: \f$ \frac{\mathrm{d}}{\mathrm{d}t} U = A\,U \f$, where \f$ A \in \mathbb{R}^{n\times n} \f$.
 * 
 * @tparam Scalar the type of scalar variables.
 */
template<typename Scalar>
class Constant : public System<Scalar> {
 public:
  using Matrix = algebra::DynamicMatrix<Scalar>;
  using Column = typename System<Scalar>::Column;

 private:
  Matrix a_;
  Column u_;

 public:
  explicit Constant(Matrix const &a) : a_(a) {
    assert(a.rows() == a_.cols());
  }

  void SetTime(double t_curr) final {
  }

  void SetSolutionColumn(Column const &u) final {
    assert(u.size() == a_.cols());
    u_ = u;
  }

  Column GetSolutionColumn() const final {
    return u_;
  }

  Column GetResidualColumn() const final {
    return a_ * u_;
  }
};

/**
 * @brief The abstract base of all ODE solvers.
 * 
 * @tparam Scalar the type of scalar variables.
 */
template <typename Scalar>
class Solver {
 public:
  using Column = typename System<Scalar>::Column;

  /**
   * @brief Update the given System from `t_curr` to `(t_curr + dt)`.
   * 
   * @param ode 
   * @param t_curr 
   * @param dt 
   */
  virtual void Update(System<Scalar> *ode, double t_curr, double dt) = 0;
};

/**
 * @brief The simplest ODE Solver: \f$ U^{n+1} = U^{n} + R(U^{n}) * \Delta t \f$.
 * 
 * @tparam Scalar the type of scalar variables.
 */
template <typename Scalar>
class Euler : public Solver<Scalar> {
 private:
  using Base = Solver<Scalar>;

 public:
  using Base::Base;
  using Column = typename Base::Column;

  static Column NextSolution(System<Scalar> *system, double t_curr, double dt) {
    system->SetTime(t_curr);
    auto u_next = system->GetSolutionColumn();
    auto residual = system->GetResidualColumn();
    residual *= dt;
    u_next += residual;
    return u_next;
  }

  void Update(System<Scalar> *system, double t_curr, double dt) final {
    auto u_next = NextSolution(system, t_curr, dt);
    system->SetSolutionColumn(u_next);
  }
};

}  // namespace temporal
}  // namespace mini

#endif  // MINI_TEMPORAL_ODE_HPP_
