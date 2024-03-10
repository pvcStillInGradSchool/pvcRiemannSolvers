// Copyright 2019 PEI Weicheng and YANG Minghao

#include <cstdlib>
#include <vector>

#include "gtest/gtest.h"

#include "mini/riemann/simple/multiple.hpp"

namespace mini {
namespace riemann {
namespace simple {

class TestRiemannSimpleMultiple : public ::testing::Test {
 protected:
  using Scalar = double;
  using Solver = Multiple<Scalar, 2, 3>;
  using Column = typename Solver::Column;
  using Matrix = typename Solver::Matrix;
  Column left{1.0, 11.0}, right{2.0, 22.0};
};
TEST_F(TestRiemannSimpleMultiple, TestTwoLeftRunningWaves) {
  // eigen_values = {-2, -1}
  auto solver = Solver(Matrix{{-2.0, 0.0}, {0.0, -1.0}});
  auto f_on_t_axis = solver.GetFluxUpwind(left, right);
  EXPECT_DOUBLE_EQ(f_on_t_axis[0], -4.0);
  EXPECT_DOUBLE_EQ(f_on_t_axis[1], -22.0);
  Matrix eigvals = solver.eigvals();
  EXPECT_EQ(solver.A() * solver.R(), solver.R() * eigvals);
  EXPECT_EQ(solver.L() * solver.A(), eigvals * solver.L());
}
TEST_F(TestRiemannSimpleMultiple, TestTwoRightRunningWaves) {
  // eigen_values = {1, 2}
  auto solver = Solver(Matrix{{1.0, 0.0}, {0.0, 2.0}});
  auto f_on_t_axis = solver.GetFluxUpwind(left, right);
  EXPECT_DOUBLE_EQ(f_on_t_axis[0], 1.0);
  EXPECT_DOUBLE_EQ(f_on_t_axis[1], 22.0);
  Matrix eigvals = solver.eigvals();
  EXPECT_EQ(solver.A() * solver.R(), solver.R() * eigvals);
  EXPECT_EQ(solver.L() * solver.A(), eigvals * solver.L());
}
TEST_F(TestRiemannSimpleMultiple, TestBetweenTwoWaves) {
  // eigen_values = {-1, 1}
  auto solver = Solver(Matrix{{-1.0, 0.0}, {0.0, +1.0}});
  auto f_on_t_axis = solver.GetFluxUpwind(left, right);
  EXPECT_DOUBLE_EQ(f_on_t_axis[0], -2.0);
  EXPECT_DOUBLE_EQ(f_on_t_axis[1], 11.0);
  Matrix eigvals = solver.eigvals();
  EXPECT_EQ(solver.A() * solver.R(), solver.R() * eigvals);
  EXPECT_EQ(solver.L() * solver.A(), eigvals * solver.L());
}
TEST_F(TestRiemannSimpleMultiple, TestNonTrivialMatrix) {
  auto solver = Solver(Matrix{{-5.0, 4.0}, {-4.0, 5.0}});
  Matrix eigvals = solver.eigvals();
  EXPECT_NEAR((solver.A() * solver.R() - solver.R() * eigvals).norm(),
      0.0, 1e-14);
  EXPECT_NEAR((solver.L() * solver.A() - eigvals * solver.L()).norm(),
      0.0, 1e-14);
  Column v_left = solver.L() * left;
  Column v_right = solver.L() * right;
  Column v_common;
  v_common[0] = eigvals(0, 0) * (eigvals(0, 0) > 0 ? v_left[0] : v_right[0]);
  v_common[1] = eigvals(1, 1) * (eigvals(1, 1) > 0 ? v_left[1] : v_right[1]);
  Column f_common = solver.R() * v_common;
  auto f_on_t_axis = solver.GetFluxUpwind(left, right);
  EXPECT_DOUBLE_EQ(f_on_t_axis[0], f_common[0]);
  EXPECT_DOUBLE_EQ(f_on_t_axis[1], f_common[1]);
}
TEST_F(TestRiemannSimpleMultiple, TestRandomMatrix) {
  std::srand(31415926);
  for (int i = 1 << 10; i > 0; --i) {
    Matrix A = Matrix::Random();
    A += A.transpose().eval();  // ensure real eigvals
    auto solver = Solver(A);
    Matrix eigvals = solver.eigvals();
    EXPECT_NEAR((A * solver.R() - solver.R() * eigvals).norm(), 0.0, 1e-14);
    EXPECT_NEAR((solver.L() * A - eigvals * solver.L()).norm(), 0.0, 1e-14);
    Column v_left = solver.L() * left;
    Column v_right = solver.L() * right;
    Column v_common;
    for (int k = 0; k < Solver::kComponents; ++k) {
      Scalar lambda = eigvals(k, k);
      v_common[k] = lambda * (lambda > 0 ? v_left[k] : v_right[k]);
    }
    Column f_common = solver.R() * v_common;
    auto f_on_t_axis = solver.GetFluxUpwind(left, right);
    EXPECT_DOUBLE_EQ(f_on_t_axis[0], f_common[0]);
    EXPECT_DOUBLE_EQ(f_on_t_axis[1], f_common[1]);
  }
}
TEST_F(TestRiemannSimpleMultiple, TestZeroMatrix) {
  auto solver = Solver(Matrix::Zero());
  auto f_on_t_axis = solver.GetFluxUpwind(left, right);
  EXPECT_DOUBLE_EQ(f_on_t_axis[0], 0.0);
  EXPECT_DOUBLE_EQ(f_on_t_axis[1], 0.0);
  Matrix eigvals = solver.eigvals();
  EXPECT_EQ(solver.A() * solver.R(), solver.R() * eigvals);
  EXPECT_EQ(solver.L() * solver.A(), eigvals * solver.L());
}

}  // namespace simple
}  // namespace riemann
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
