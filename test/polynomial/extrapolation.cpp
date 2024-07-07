//  Copyright 2024 PEI Weicheng

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
  using Projection = typename Extrapolation::Projection;
  auto extrapolation = Extrapolation(integrator);
  // test SetCoeff() and SetValue()
  extrapolation.SetZero();
  EXPECT_EQ(extrapolation.coeff(),
      Extrapolation::Coeff::Zero());
  EXPECT_EQ(extrapolation.interpolation().coeff(),
      Interpolation::Coeff::Zero());
  EXPECT_EQ(extrapolation.projection().coeff(),
      Projection::Coeff::Zero());
  Extrapolation another_extrapolation = extrapolation;
  typename Extrapolation::Coeff coeff;
  for (int ijk = 0; ijk < Extrapolation::N; ++ijk) {
    Value value = Value::Random();
    coeff.col(ijk) = value;
    EXPECT_NE(extrapolation.coeff(), coeff);
    extrapolation.SetCoeff(ijk, value);
    another_extrapolation.SetValue(ijk, value);
    EXPECT_NEAR((another_extrapolation.GetValue(ijk) - value).norm(),
        0.0, 1e-15);
  }
  EXPECT_EQ(extrapolation.coeff(), coeff);
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
  auto interp = [&](Global const &global) {
    return extrapolation.GlobalToValue(global);
  };
  auto extrap = [&](Global const &global) {
    return extrapolation.Extrapolate(global);
  };
  std::cout << "norm(exact - interp) =\n"
      << mini::integrator::Distance(exact, interp, integrator) << std::endl;
  std::cout << "norm(exact - extrap) =\n"
      << mini::integrator::Distance(exact, extrap, integrator) << std::endl;
  std::cout << "norm(extrap - interp) =\n"
      << mini::integrator::Distance(interp, extrap, integrator) << std::endl;
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
