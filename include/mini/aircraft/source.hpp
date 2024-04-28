// Copyright 2022 PEI Weicheng
#ifndef MINI_AIRCRAFT_SOURCE_HPP_
#define MINI_AIRCRAFT_SOURCE_HPP_

#include <concepts>

#include <algorithm>
#include <type_traits>
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
template <typename P, std::floating_point Scalar>
class Rotorcraft {
 public:
  using Part = P;
  using Cell = typename Part::Cell;
  using Face = typename Cell::Face;
  static_assert(std::is_same_v<Scalar, typename Cell::Scalar>);
  using Global = typename Cell::Global;
  using Coeff = typename Cell::Projection::Coeff;
  using Conservative = typename Part::Riemann::Conservative;
  using Rotor = mini::aircraft::Rotor<Scalar>;
  using Blade = typename Rotor::Blade;
  using Section = typename Blade::Section;
  using Force = Global;

 private:
  static bool Valid(Scalar ratio) {
    return 0.0 <= ratio && ratio <= 1.0;
  }

  static std::pair<Scalar, Scalar>
  Intersect(const Cell &cell, const Blade &blade) {
    const Global &p = blade.P();
    const Global &q = blade.Q();
    const Global &pq = blade.PQ();
    Scalar r_ratio{-1}, s_ratio{-1};
    for (const Face *face : cell.adj_faces_) {
      if (Valid(r_ratio) && Valid(s_ratio)) {
        break;
      }
      const auto &gauss = face->gauss();
      // Currently, only triangle is supported.
      assert(gauss.coordinate().CountCorners() == 3);
      Global pa = gauss.coordinate().GetGlobal(0) - p;
      Global pb = gauss.coordinate().GetGlobal(1) - p;
      Global pc = gauss.coordinate().GetGlobal(2) - p;
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

  static std::pair<Force, Scalar>
  GetSourceValue(const Cell &cell, const Section &section, const Global &xyz) {
    auto value = cell.GlobalToValue(xyz);
    auto &cv = *reinterpret_cast<Conservative *>(&value);
    auto uvw = cv.momentum() / cv.mass();
    Force force = section.GetForce(cv.mass(), uvw);
    auto power = force.transpose() * uvw;
    return {force, power};
  }

  void UpdateCoeff(const Cell &cell, const Blade &blade, Scalar *coeff_data) {
    auto [r_ratio, s_ratio] = Intersect(cell, blade);
    if (r_ratio < s_ratio) {
      // r_ratio is always set before s_ratio
      assert(Valid(r_ratio) && Valid(s_ratio));
      // Integrate along RS:
      auto line = mini::integrator::Line<Scalar, 1, 4>(r_ratio, s_ratio);
      auto integrand = [&cell, &blade](Scalar ratio){
        auto section = blade.GetSection(ratio);
        auto xyz = section.GetOrigin();
        auto [force, power] = GetSourceValue(cell, section, xyz);
        using Mat1xN = mini::algebra::Matrix<Scalar, 1, Cell::N>;
        Mat1xN basis_values = cell.GlobalToBasisValues(xyz);
        using Mat4xN = mini::algebra::Matrix<Scalar, 4, Cell::N>;
        Mat4xN product;
        product.row(0) = force[0] * basis_values;
        product.row(1) = force[1] * basis_values;
        product.row(2) = force[2] * basis_values;
        product.row(3) = power * basis_values;
        return product;
      };
      auto integral = mini::integrator::Integrate(integrand, line);
      integral *= blade.GetSpan();
      auto *coeff = reinterpret_cast<Coeff *>(coeff_data);
      coeff->row(1) += integral.row(0);
      coeff->row(2) += integral.row(1);
      coeff->row(3) += integral.row(2);
      coeff->row(4) += integral.row(3);
    }
  }

 public:
  void UpdateCoeff(const Cell &cell, double t_curr, Scalar *coeff_data) {
    for (auto &rotor : rotors_) {
      rotor.UpdateAzimuth(t_curr);
      for (int i = 0, n = rotor.CountBlades(); i < n; ++i) {
        UpdateCoeff(cell, rotor.GetBlade(i), coeff_data);
      }
    }
  }

  void GetForces(const Cell &cell, double t_curr, std::vector<Global> *forces,
      std::vector<Global> *points, std::vector<Scalar> *weights) {
    for (auto &rotor : rotors_) {
      rotor.UpdateAzimuth(t_curr);
      for (int i = 0, n = rotor.CountBlades(); i < n; ++i) {
        const Blade &blade = rotor.GetBlade(i);
        auto [r_ratio, s_ratio] = Intersect(cell, blade);
        if (r_ratio >= s_ratio) {
          continue;
        }
        constexpr int Q = 4;
        auto line = mini::integrator::Line<Scalar, 1, Q>(r_ratio, s_ratio);
        for (int q = 0; q < Q; ++q) {
          auto ratio = line.GetGlobal(q);
          auto section = blade.GetSection(ratio);
          auto xyz = section.GetOrigin();
          auto [force, power] = GetSourceValue(cell, section, xyz);
          forces->emplace_back(force);
          points->emplace_back(xyz);
          weights->emplace_back(line.GetGlobalWeight(q) * blade.GetSpan());
        }
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
