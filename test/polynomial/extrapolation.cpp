//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cstdlib>
#include <cmath>

#include "mini/integrator/function.hpp"
#include "mini/integrator/legendre.hpp"
#include "mini/integrator/lobatto.hpp"
#include "mini/integrator/quadrangle.hpp"
#include "mini/integrator/hexahedron.hpp"
#include "mini/coordinate/quadrangle.hpp"
#include "mini/coordinate/hexahedron.hpp"
#include "mini/basis/linear.hpp"
#include "mini/polynomial/projection.hpp"
#include "mini/polynomial/hexahedron.hpp"
#include "mini/polynomial/extrapolation.hpp"
#include "mini/constant/index.hpp"

#include "gtest/gtest.h"

using std::sqrt;
using namespace mini::constant::index;

double rand_f() {
  return -10 + 2 * std::rand() / (1.0 + RAND_MAX);
}

class TestPolynomialExtrapolation : public ::testing::Test {
 protected:
  using Scalar = double;
  static constexpr int kComponents = 3;
  static constexpr int kDegrees = 3;
};
TEST_F(TestPolynomialExtrapolation, Hexahedron) {
  // build a hexa Coordinate
  using Coordinate = mini::coordinate::Hexahedron8<Scalar>;
  using Global = typename Coordinate::Global;
  auto coordinate = Coordinate {
    Global(-10, -10, -10), Global(+10, -10, -10),
    Global(+10, +10, -10), Global(-10, +10, -10),
    Global(-10, -10, +10), Global(+10, -10, +10),
    Global(+10, +10, +10), Global(-10, +10, +10),
  };
  // build a hexa Integrator
  using Gx = mini::integrator::Lobatto<Scalar, kDegrees + 1>;
  using Integrator = mini::integrator::Hexahedron<Gx, Gx, Gx>;
  auto integrator = Integrator(coordinate);
  // build a hexa Extrapolation
  using Interpolation = mini::polynomial::Hexahedron<
      Gx, Gx, Gx, kComponents, true>;
  static_assert(std::is_same_v<Integrator, typename Interpolation::Integrator>);
  using Value = typename Interpolation::Value;
  using Extrapolation = mini::polynomial::Extrapolation<Interpolation>;
  auto extrapolation = Extrapolation(integrator);
  // approximate a polynomial function
  auto exact = [](Global const &global) -> Value {
    using mini::constant::index::X;
    using mini::constant::index::Y;
    using mini::constant::index::Z;
    Value value;
    value[X] = std::sin(global[X]);
    value[Y] = std::sin(global[Y]);
    value[Z] = std::sin(global[Z]);
    return value;
  };
  extrapolation.Approximate(exact);
  auto norm = mini::integrator::Distance(exact,
      [&](Global const &global){ return extrapolation.GlobalToValue(global); },
      integrator);
  std::cout << "norm(exact - interpolation) =\n"
      << norm.transpose() << std::endl;
  norm = mini::integrator::Distance(exact,
      [&](Global const &global){ return extrapolation.Extrapolate(global); },
      integrator);
  std::cout << "norm(exact - extrapolation) =\n"
      << norm.transpose() << std::endl;
  norm = mini::integrator::Distance(
      [&](Global const &global){ return extrapolation.Extrapolate(global); },
      [&](Global const &global){ return extrapolation.GlobalToValue(global); },
      integrator);
  std::cout << "norm(extrapolation - interpolation) =\n"
      << norm.transpose() << std::endl;
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
