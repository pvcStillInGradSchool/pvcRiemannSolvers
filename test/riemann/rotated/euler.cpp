// Copyright 2019 PEI Weicheng and YANG Minghao

#include <cstdlib>
#include <vector>

#include "gtest/gtest.h"

#include "mini/geometry/pi.hpp"
#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/exact.hpp"

#define SOLVE_RIEMANN_PROBLEM_ON_INVISCID_WALL_
#include "mini/riemann/rotated/euler.hpp"

namespace mini {
namespace riemann {
namespace rotated {

class TestRotatedEuler : public ::testing::Test {
 protected:
  using Gas = euler::IdealGas<double, 1.4>;
  static void ExpectNear(double x, double y, double eps) {
    if (x == 0) {
      EXPECT_EQ(y, 0);
    } else {
      EXPECT_NEAR(y / x, 1, eps);
    }
  }
  static double rand_f() {
    return -1 + 2.0 * std::rand() / (1.0 + RAND_MAX);
  }
  static double disturb(double x) {
    return x * (1 + rand_f() * 0.05);
  }
};
TEST_F(TestRotatedEuler, Test2dConverter) {
  using UnrotatedSolver = euler::Exact<Gas, 2>;
  using Solver = Euler<UnrotatedSolver>;
  using Vector = Solver::Vector;
  using Frame = Solver::Frame;
  using Value = Solver::Value;
  Solver solver;
  Frame frame { Vector{+0.6, 0.8}, Vector{-0.8, 0.6} };
  Value v{0, 3.0, 4.0, 0}, v_copy{0, 3.0, 4.0, 0};
  solver.Rotate(frame);
  solver.GlobalToNormal(&v);
  Vector momentum = v_copy.momentum();
  EXPECT_EQ(v.momentumX(), momentum.dot(frame[0]));
  EXPECT_NEAR(v.momentumY(), momentum.dot(frame[1]), 1e-15);
  solver.NormalToGlobal(&v);
  EXPECT_DOUBLE_EQ(v[0], v_copy[0]);
  EXPECT_DOUBLE_EQ(v[1], v_copy[1]);
  EXPECT_DOUBLE_EQ(v[2], v_copy[2]);
  EXPECT_DOUBLE_EQ(v[3], v_copy[3]);
}
TEST_F(TestRotatedEuler, Test3dConverter) {
  using Solver = Euler<euler::Exact<Gas, 3>>;
  using Scalar = Solver::Scalar;
  using Vector = Solver::Vector;
  using Frame = Solver::Frame;
  using Value = Solver::Value;
  Solver solver;
  Frame frame {
    Vector{+0.6, 0.8, 0.0}, Vector{-0.8, 0.6, 0.0}, Vector{0.0, 0.0, 1.0}
  };
  Value v{0, 3.0, 4.0, 5.0, 0}, v_copy{0, 3.0, 4.0, 5.0, 0};
  solver.Rotate(frame);
  solver.GlobalToNormal(&v);
  Vector momentum = v_copy.momentum();
  EXPECT_EQ(v.momentumX(), momentum.dot(frame[0]));
  EXPECT_EQ(v.momentumY(), momentum.dot(frame[1]));
  EXPECT_EQ(v.momentumZ(), momentum.dot(frame[2]));
  solver.NormalToGlobal(&v);
  EXPECT_DOUBLE_EQ(v[0], v_copy[0]);
  EXPECT_DOUBLE_EQ(v[1], v_copy[1]);
  EXPECT_DOUBLE_EQ(v[2], v_copy[2]);
  EXPECT_DOUBLE_EQ(v[3], v_copy[3]);
  EXPECT_DOUBLE_EQ(v[4], v_copy[4]);
}
TEST_F(TestRotatedEuler, Test3dSolver) {
  using Solver = Euler<euler::Exact<Gas, 3>>;
  using Vector = Solver::Vector;
  using Scalar = Solver::Scalar;
  using Primitive = Solver::Primitive;
  using Speed = Scalar;
  using Flux = Solver::Flux;

  auto CompareFlux = [](Flux const& lhs, Flux const& rhs) {
    constexpr double eps = 1e-4;
    ExpectNear(lhs.mass(), rhs.mass(), eps);
    ExpectNear(lhs.energy(), rhs.energy(), eps);
    ExpectNear(lhs.momentumX(), rhs.momentumX(), eps);
    ExpectNear(lhs.momentumY(), rhs.momentumY(), eps);
    ExpectNear(lhs.momentumZ(), rhs.momentumZ(), eps);
  };

  Solver solver;
  auto frame = std::array<Vector, 3>();
  Speed v__left{1.5}, v_right{2.5};
  Speed w__left{1.5}, w_right{0.5};

  frame[0] = { 1, 0, 0 };
  frame[1] = { 0, 1, 0 };
  frame[2] = { 0, 0, 1 };
  solver.Rotate(frame);
  Primitive  left{1.000, 0.0, v__left, w__left, 1.0};
  Primitive right{0.125, 0.0, v_right, w_right, 0.1};
  auto left_c  = Gas::PrimitiveToConservative(left);
  auto right_c = Gas::PrimitiveToConservative(right);
  CompareFlux(solver.GetFluxUpwind(left_c, right_c),
      solver.GetFlux({0.426319, +0.927453, v__left, w__left, 0.303130}));
  CompareFlux(solver.GetFluxUpwind(right_c, left_c),
      solver.GetFlux({0.426319, -0.927453, v__left, w__left, 0.303130}));

  frame[1] = { 0, -1, 0 };
  frame[2] = { 0, 0, -1 };
  solver.Rotate(frame);
  CompareFlux(solver.GetFluxUpwind(left_c, right_c),
      solver.GetFlux({0.426319, +0.927453, v__left, w__left, 0.303130}));
  CompareFlux(solver.GetFluxUpwind(right_c, left_c),
      solver.GetFlux({0.426319, -0.927453, v__left, w__left, 0.303130}));

  std::srand(31415926);
  for (int i = 1 << 10; i > 0; --i) {
    Scalar angle = 180 * rand_f();
    auto [cos, sin] = mini::geometry::CosSin(angle);
    frame[1] = { 0, cos, +sin };
    frame[2] = { 0, -sin, cos };
    solver.Rotate(frame);
    Scalar rho = disturb(1.29);
    Scalar u = 10 * rand_f();
    Scalar v = 20 * rand_f();
    Scalar w = 30 * rand_f();
    Scalar p = disturb(101325);
    Primitive left{rho, u, v, w, p};
    auto left_c  = Gas::PrimitiveToConservative(left);
    auto right_c = left_c;
    right_c.momentumX() = -left_c.momentumX();
    auto flux = solver.GetFluxOnInviscidWall(left_c);
#ifdef SOLVE_RIEMANN_PROBLEM_ON_INVISCID_WALL_
    CompareFlux(solver.GetFluxUpwind(left_c, right_c), flux);
    EXPECT_NEAR(flux.momentumX() / p, 1.0, 5e-2);
#else
    EXPECT_NEAR(p, flux.momentumX(), 1e-10);
#endif
    EXPECT_EQ(0.0, flux.mass());
    EXPECT_EQ(0.0, flux.momentumY());
    EXPECT_EQ(0.0, flux.momentumZ());
    EXPECT_EQ(0.0, flux.energy());
  }
}

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
