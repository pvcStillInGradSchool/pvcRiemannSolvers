// Copyright 2022 PEI Weicheng
#ifndef MINI_GEOMETRY_PI_HPP_
#define MINI_GEOMETRY_PI_HPP_

#include <cmath>
#include <numbers>
#include <utility>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace geometry {

inline constexpr double pi() {
  return std::numbers::pi_v<double>;
  return 3.1415926535897932384626433832795028841971693993751;
}
inline constexpr double deg2rad(double deg) {
  return deg * pi() / 180;
}
inline constexpr double rad2deg(double rad) {
  return rad / pi() * 180;
}
inline constexpr std::pair<double, double> CosSin(double deg) {
  auto rad = deg2rad(deg);
  return { std::cos(rad), std::sin(rad) };
}

}  // namespace geometry
}  // namespace mini

#endif  // MINI_GEOMETRY_PI_HPP_
