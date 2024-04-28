//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_GAUSS_FACE_HPP_
#define MINI_GAUSS_FACE_HPP_

#include <concepts>

#include "mini/coordinate/face.hpp"
#include "mini/integrator/element.hpp"
#include "mini/integrator/function.hpp"

namespace mini {
namespace integrator {

/**
 * @brief Abstract numerical integrators on surface elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam kPhysDim  Dimension of the physical space.
 */
template <std::floating_point Scalar, int kPhysDim>
class Face : public Element<Scalar, kPhysDim, 2> {
  static constexpr int D = kPhysDim;
  static_assert(D == 2 || D == 3);

 public:
  using Coordinate = coordinate::Face<Scalar, kPhysDim>;
  using Real = typename Coordinate::Real;
  using Local = typename Coordinate::Local;
  using Global = typename Coordinate::Global;
  using Jacobian = typename Coordinate::Jacobian;
  using Frame = typename Coordinate::Frame;

  virtual ~Face() noexcept = default;
  virtual const Frame &GetNormalFrame(int i) const = 0;
  virtual Scalar area() const = 0;

  /**
   * @brief Get a reference to the coordinate::Face object it uses for coordinate mapping.
   * 
   * @return const Coordinate &  Reference to the coordinate::Face object it uses for coordinate mapping.
   */
  virtual const Coordinate &coordinate() const = 0;

 protected:
  static void BuildNormalFrames(Face *face) requires(kPhysDim == 2) {
  }
  static void BuildNormalFrames(Face *face) requires(kPhysDim == 3) {
    int n = face->CountPoints();
    for (int i = 0; i < n; ++i) {
      auto &local_i = face->GetLocal(i);
      face->GetNormalFrame(i) = face->coordinate().LocalToNormalFrame(local_i);
    }
  }
  virtual Frame &GetNormalFrame(int i) = 0;
};

}  // namespace integrator
}  // namespace mini

#endif  // MINI_GAUSS_FACE_HPP_
