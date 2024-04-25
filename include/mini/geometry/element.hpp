//  Copyright 2023 PEI Weicheng
#ifndef MINI_GEOMETRY_ELEMENT_HPP_
#define MINI_GEOMETRY_ELEMENT_HPP_

#include <concepts>

#include <cassert>
#include <cmath>
#include <numeric>

#include <initializer_list>
#include <iostream>
#include <vector>

#include "mini/algebra/eigen.hpp"
#include "mini/constant/index.hpp"

namespace mini {
namespace geometry {

using namespace mini::constant::index;

/**
 * @brief Abstract coordinate map on surface elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam kPhysDim  Dimension of the underlying physical space.
 * @tparam kCellDim  Dimension of the element as a manifold.
 */
template <std::floating_point Scalar, int kPhysDim, int kCellDim>
class Element {
 public:
  using Real = Scalar;
  using Local = algebra::Matrix<Scalar, kCellDim, 1>;
  using Global = algebra::Matrix<Scalar, kPhysDim, 1>;

  /**
   * @brief The type of (geometric) Jacobian matrix, which is defined as \f$ \mathbf{J}=\begin{bmatrix}\partial_{\xi}\\\partial_{\eta}\\\partial_{\zeta}\end{bmatrix}\begin{bmatrix}x & y & z\end{bmatrix} \f$.
   * 
   */
  using Jacobian = algebra::Matrix<Scalar, kCellDim, kPhysDim>;

  static constexpr int CellDim() { return kCellDim; }
  static constexpr int PhysDim() { return kPhysDim; }

  virtual ~Element() noexcept = default;
  virtual std::vector<Scalar> LocalToShapeFunctions(const Local &) const = 0;

  /**
   * @brief \f$ \begin{bmatrix}\partial_{\xi}\\ \partial_{\eta}\\ \cdots \end{bmatrix}\begin{bmatrix}\phi_{1} & \phi_{2} & \cdots\end{bmatrix}=\begin{bmatrix}\partial_{\xi}\phi_{1} & \partial_{\xi}\phi_{2} & \cdots\\ \partial_{\eta}\phi_{1} & \partial_{\eta}\phi_{2} & \cdots\\ \cdots & \cdots & \cdots \end{bmatrix} \f$
   * 
   * @return std::vector<Local> 
   */
  virtual std::vector<Local> LocalToShapeGradients(const Local &) const = 0;
  virtual Global LocalToGlobal(const Local &) const = 0;
  virtual Jacobian LocalToJacobian(const Local &) const = 0;
  virtual int CountCorners() const = 0;
  virtual int CountNodes() const = 0;
  virtual const Local &GetLocalCoord(int i) const = 0;
  virtual const Global &GetGlobalCoord(int i) const = 0;
  virtual const Global &center() const = 0;

 protected:
  virtual void _BuildCenter() = 0;
  Global &_GetGlobalCoord(int i) {
    const Global &global
        = const_cast<const Element *>(this)->GetGlobalCoord(i);
    return const_cast<Global &>(global);
  }

  static void _Build(Element *element,
      std::initializer_list<Global> il) {
    assert(il.size() == element->CountNodes());
    auto p = il.begin();
    for (int i = 0, n = element->CountNodes(); i < n; ++i) {
      element->_GetGlobalCoord(i) = p[i];
    }
    element->_BuildCenter();
  }

  /**
   * @brief Mimic [`scipy.optimize.root`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html).
   * 
   * @tparam Func 
   * @tparam MatJ 
   * @param func 
   * @param x 
   * @param matj 
   * @param xtol 
   * @return requires&& 
   */
  template <typename Func, typename MatJ>
      requires std::is_same_v<Global, std::invoke_result_t<Func, Local const &>>
          && std::is_same_v<Jacobian, std::invoke_result_t<MatJ, Local const &>>
  static Global root(Func &&func, Global x, MatJ &&matj, Scalar xtol = 1e-5,
      Scalar max_res_norm = 0.5, int cnt = 128) requires(kCellDim == kPhysDim) {
    Global res;
    Scalar res_norm;
#ifndef NDEBUG
    std::vector<Local> x_history;
    x_history.reserve(cnt);
    x_history.emplace_back(x);
#endif
    do {
      /**
       * The Jacobian matrix required here is the transpose of the one returned by `Element::LocalToJacobian`.
       */
      res = matj(x).transpose().partialPivLu().solve(func(x));
      res_norm = res.norm();
      if (res_norm > max_res_norm) {
        res *= (max_res_norm / res_norm);
        res_norm = max_res_norm;
      }
      x -= res;
      cnt--;
#ifndef NDEBUG
      x_history.emplace_back(x);
#endif
    } while (cnt && res_norm > xtol);
    if (cnt == 0 || std::isnan(res_norm)) {
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

 public:
  Local GlobalToLocal(const Global &global,
      const Local &hint = Local::Zero()) const requires(kCellDim == kPhysDim) {
    auto func = [this, &global](Local const &local) {
      auto res = LocalToGlobal(local);
      return res -= global;
    };
    auto jac = [this](Local const &local) {
      return LocalToJacobian(local);
    };
    Local local;
    try {
      local = root(func, hint, jac);
    } catch(std::runtime_error &e) {
      std::cerr << "global = " << global.transpose() << "\n";
      std::cerr << "global_coords =" << "\n";
      for (int i = 0; i < this->CountNodes(); ++i) {
        std::cerr << this->GetGlobalCoord(i).transpose() << "\n";
      }
      std::cerr << std::endl;
      throw e;
    }
    return local;
  }
};

}  // namespace geometry
}  // namespace mini

#endif  // MINI_GEOMETRY_ELEMENT_HPP_
