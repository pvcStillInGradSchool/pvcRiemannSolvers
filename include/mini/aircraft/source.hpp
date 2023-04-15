// Copyright 2022 PEI Weicheng
#ifndef MINI_AIRCRAFT_SOURCE_HPP_
#define MINI_AIRCRAFT_SOURCE_HPP_

#include <algorithm>
#include <utility>
#include <vector>

#include "mini/algebra/eigen.hpp"
#include "mini/geometry/frame.hpp"
#include "mini/geometry/intersect.hpp"
#include "mini/integrator/line.hpp"
#include "mini/aircraft/rotor.hpp"

namespace mini {
namespace aircraft {

/**
 * @brief A rotorcraft is an aircraft that has multiple rotors.
 * 
 * @tparam P 
 * @tparam Scalar 
 */
template <typename P, typename Scalar>
class Rotorcraft {
 public:
  using Part = P;
  using Cell = typename Part::Cell;
  using Face = typename Cell::Face;
  using Coord = typename Cell::Coord;
  using Coeff = typename Cell::Projection::Coeff;
  using Conservative = typename Part::Riemann::Conservative;
  using Rotor = mini::aircraft::Rotor<Scalar>;
  using Blade = typename Rotor::Blade;

 private:
  static bool Valid(Scalar ratio) {
    return 0.0 <= ratio && ratio <= 1.0;
  }

  static std::pair<Scalar, Scalar>
  Intersect(const Cell &cell, const Blade &blade) {
    const Coord &p = blade.P();
    const Coord &q = blade.Q();
    const Coord &pq = blade.PQ();
    Scalar r_ratio{-1}, s_ratio{-1};
    for (const Face *face : cell.adj_faces_) {
      if (Valid(r_ratio) && Valid(s_ratio)) {
        break;
      }
      const auto &gauss = face->gauss();
      // Currently, only triangle is supported.
      assert(gauss.CountVertices() == 3);
      Coord pa = gauss.GetVertex(0) - p;
      Coord pb = gauss.GetVertex(1) - p;
      Coord pc = gauss.GetVertex(2) - p;
      Scalar ratio = -1.0;
      mini::geometry::Intersect(pa, pb, pc, pq, &ratio);
      if (Valid(ratio)) {
        if (!Valid(r_ratio)) {
          r_ratio = ratio;
        } else if (!Valid(s_ratio)) {
          s_ratio = ratio;
        } else {
          // More than two common points are found.
          assert(false);
        }
      }
    }
    if (Valid(r_ratio) && !Valid(s_ratio)) {
      // If only one common point is found (R is always found before S),
      // then either P or Q is inside.
      s_ratio = r_ratio < 0.5 ? 0 : 1;  // p_ratio = 0, q_ratio = 1
    }
    if (Valid(r_ratio) && Valid(s_ratio)) {
      if (r_ratio > s_ratio) {
        std::swap(r_ratio, s_ratio);
      }
    }
    return {r_ratio, s_ratio};
  }

  void UpdateCoeff(const Cell &cell, const Blade &blade, Coeff *coeff) {
    auto [r_ratio, s_ratio] = Intersect(cell, blade);
    if (r_ratio < s_ratio) {
      // r_ratio is always set before s_ratio
      assert(Valid(r_ratio) && Valid(s_ratio));
      // Integrate along RS:
      auto line = mini::integrator::Line<Scalar, 1, 4>(r_ratio, s_ratio);
      auto func = [&cell, &blade](Scalar ratio){
        auto section = blade.GetSection(ratio);
        auto xyz = section.GetOrigin();
        const auto &proj = cell.projection_;
        auto value = proj(xyz);  // conservative variable
        auto &cv = *reinterpret_cast<Conservative *>(&value);
        auto uvw = cv.momentum() / cv.mass();
        auto force = section.GetForce(cv.mass(), uvw);
        using Mat1xN = mini::algebra::Matrix<Scalar, 1, Cell::N>;
        Mat1xN basis_values = cell.basis_(xyz).transpose();
        using Mat4xN = mini::algebra::Matrix<Scalar, 4, Cell::N>;
        Mat4xN prod;
        prod.row(0) = force[0] * basis_values;
        prod.row(1) = force[1] * basis_values;
        prod.row(2) = force[2] * basis_values;
        auto work = force.transpose() * uvw;
        prod.row(3) = work * basis_values;
        return prod;
      };
      auto integral = mini::integrator::Integrate(func, line);
      integral *= blade.GetSpan();
      coeff->row(1) += integral.row(0);
      coeff->row(2) += integral.row(1);
      coeff->row(3) += integral.row(2);
      coeff->row(4) += integral.row(3);
    }
  }

 public:
  void UpdateCoeff(const Cell &cell, double t_curr, Coeff *coeff) {
    for (auto &rotor : rotors_) {
      rotor.UpdateAzimuth(t_curr);
      for (int i = 0, n = rotor.CountBlades(); i < n; ++i) {
        UpdateCoeff(cell, rotor.GetBlade(i), coeff);
      }
    }
  }

  Rotorcraft &InstallRotor(const Rotor &rotor) {
    rotors_.emplace_back(rotor);
    return *this;
  }

 protected:
  std::vector<Rotor> rotors_;
};

}  // namespace aircraft
}  // namespace mini

#endif  // MINI_AIRCRAFT_SOURCE_HPP_
