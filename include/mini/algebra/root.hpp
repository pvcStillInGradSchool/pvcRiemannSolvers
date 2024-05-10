//  Copyright 2024 PEI Weicheng
#ifndef MINI_ALGEBRA_ROOT_HPP_
#define MINI_ALGEBRA_ROOT_HPP_

#include <vector>
#include <iostream>
#include <type_traits>
#include <vector>

namespace mini {
namespace algebra {
namespace root {

/**
 * @brief Solve a non-linear algebra equation (system) \f$ F(X) = O \f$ by Newton's iteration.
 * 
 * Mimic [`scipy.optimize.root`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html).
 * 
 * @tparam X the type of \f$ X \f$, which must be a column
 * @tparam XtoColumn the type of \f$ F \f$, which returns a column
 * @tparam XtoJacobian the type of \f$ \partial F / \partial X \f$
 * @param x 
 * @param x_to_column 
 * @param x_to_jacobian 
 * @param x_update_norm_tolerance 
 * @param max_update_norm 
 * @param remaining_updates 
 * @return 
 */
template <typename X, typename XtoColumn, typename XtoJacobian>
X Newton(X x, XtoColumn &&x_to_column, XtoJacobian &&x_to_jacobian,
    double x_update_norm_tolerance = 1e-5, double max_update_norm = 0.5,
    int remaining_updates = 128) {
  using Column = std::invoke_result_t<XtoColumn, X const &>;
  using Jacobian = std::invoke_result_t<XtoJacobian, X const &>;
  X x_update;
  double x_update_norm;
#ifndef NDEBUG
  std::vector<X> x_history;
  x_history.reserve(remaining_updates);
  x_history.emplace_back(x);
#endif
  do {
    /**
     * The Jacobian matrix required here is the transpose of the one returned by `Element::LocalToJacobian`.
     */
    Jacobian jacobian = x_to_jacobian(x).transpose();
    x_update = jacobian.partialPivLu().solve(x_to_column(x));
    x_update_norm = x_update.norm();
    if (x_update_norm > max_update_norm) {
      x_update *= (max_update_norm / x_update_norm);
      x_update_norm = max_update_norm;
    }
    x -= x_update;
    remaining_updates--;
#ifndef NDEBUG
    x_history.emplace_back(x);
#endif
  } while (remaining_updates && x_update_norm > x_update_norm_tolerance);
  if (remaining_updates == 0 || std::isnan(x_update_norm)) {
#ifndef NDEBUG
    std::cerr << "x_history =\n";
    for (int i = 0; i < x_history.size(); ++i) {
      std::cerr << x_history[i].transpose() << "\n";
    }
    std::cerr << std::endl;
#endif
    throw std::runtime_error("Exceed maximum iteration steps.");
  }
  return x;
}

}  // namespace root
}  // namespace algebra
}  // namespace mini

#endif  // MINI_ALGEBRA_ROOT_HPP_
