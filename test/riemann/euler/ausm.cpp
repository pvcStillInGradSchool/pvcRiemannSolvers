// Copyright 2019 PEI Weicheng and YANG Minghao

#include <vector>

#include "gtest/gtest.h"

#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/ausm.hpp"

namespace mini {
namespace riemann {
namespace euler {

double rand_f() {
  return std::rand() / (1.0 + RAND_MAX);
}

double ratio(double x, double y) {
  return std::abs(x - y) / std::max(std::abs(x), std::abs(y));
}

class TestAdvectionUpstreamSplittingMethod : public ::testing::Test {
 protected:
  using Gas = IdealGas<double, 1.4>;
  using Solver = AdvectionUpstreamSplittingMethod<Gas, 1>;
  using Primitive = Solver::Primitive;
  using Flux = Solver::Flux;
  Solver solver;
  static void CompareFlux(Flux const& lhs, Flux const& rhs) {
    EXPECT_LE(ratio(lhs.mass(), rhs.mass()), 0.34);
    EXPECT_LE(ratio(lhs.energy(), rhs.energy()), 0.19);
    EXPECT_LE(ratio(lhs.momentumX(), rhs.momentumX()), 0.27);
  }
};
TEST_F(TestAdvectionUpstreamSplittingMethod, TestConsistency) {
  auto rho{0.1}, u{0.2}, p{0.3};
  auto flux = Flux{rho * u, rho * u * u + p, u};
  flux.energy() *= p * Gas::GammaOverGammaMinusOne() + 0.5 * rho * u * u;
  EXPECT_EQ(solver.GetFlux({rho, u, p}), flux);
  std::srand(31415926);
  Primitive state{rand_f(), rand_f(), rand_f()};
  Flux diff = solver.GetFluxUpwind(state, state) - solver.GetFlux(state);
  EXPECT_NEAR(diff.norm(), 0.0, 1e-15);
}
TEST_F(TestAdvectionUpstreamSplittingMethod, TestSod) {
  Primitive left{1.0, 0.0, 1.0}, right{0.125, 0.0, 0.1};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.426319, +0.927453, 0.303130}));
  CompareFlux(solver.GetFluxUpwind(right, left),
              solver.GetFlux({0.426319, -0.927453, 0.303130}));
}
TEST_F(TestAdvectionUpstreamSplittingMethod, TestShockCollision) {
  Primitive left{5.99924, 19.5975, 460.894}, right{5.99242, 6.19633, 46.0950};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({5.99924, 19.5975, 460.894}));
}
TEST_F(TestAdvectionUpstreamSplittingMethod, TestBlastFromLeft) {
  Primitive left{1.0, 0.0, 1000}, right{1.0, 0.0, 0.01};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.575062, 19.59745, 460.8938}));
}
TEST_F(TestAdvectionUpstreamSplittingMethod, TestBlastFromRight) {
  Primitive left{1.0, 0.0, 0.01}, right{1.0, 0.0, 100};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.575113, -6.196328, 46.09504}));
}

class AdvectionUpstreamSplittingMethod2dTest : public ::testing::Test {
 protected:
  using Solver = AdvectionUpstreamSplittingMethod<IdealGas<double, 1.4>, 2>;
  using Primitive = Solver::Primitive;
  using Speed = Solver::Scalar;
  using Flux = Solver::Flux;
  Solver solver;
  Speed v__left{1.5}, v_right{2.5};
  static void CompareFlux(Flux const& lhs, Flux const& rhs) {
    EXPECT_LE(ratio(lhs.mass(), rhs.mass()), 0.34);
    EXPECT_LE(ratio(lhs.energy(), rhs.energy()), 0.27);
    EXPECT_LE(ratio(lhs.momentumX(), rhs.momentumX()), 0.27);
    EXPECT_LE(ratio(lhs.momentumY(), rhs.momentumY()), 0.40);
  }
};
TEST_F(AdvectionUpstreamSplittingMethod2dTest, TestSod) {
  Primitive  left{1.000, 0.0, v__left, 1.0};
  Primitive right{0.125, 0.0, v_right, 0.1};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.426319, +0.927453, v__left, 0.303130}));
  CompareFlux(solver.GetFluxUpwind(right, left),
              solver.GetFlux({0.426319, -0.927453, v__left, 0.303130}));
}
TEST_F(AdvectionUpstreamSplittingMethod2dTest, TestShockCollision) {
  Primitive  left{5.99924, 19.5975, v__left, 460.894};
  Primitive right{5.99242, 6.19633, v_right, 46.0950};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({5.99924, 19.5975, v__left, 460.894}));
}
TEST_F(AdvectionUpstreamSplittingMethod2dTest, TestBlastFromLeft) {
  Primitive  left{1.0, 0.0, v__left, 1e+3};
  Primitive right{1.0, 0.0, v_right, 1e-2};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.575062, 19.59745, v__left, 460.8938}));
}
TEST_F(AdvectionUpstreamSplittingMethod2dTest, TestBlastFromRight) {
  Primitive  left{1.0, 0.0, v__left, 1e-2};
  Primitive right{1.0, 0.0, v_right, 1e+2};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.575113, -6.196328, v_right, 46.09504}));
}

}  // namespace euler
}  // namespace riemann
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
