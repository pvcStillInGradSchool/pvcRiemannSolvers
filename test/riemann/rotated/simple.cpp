// Copyright 2024 PEI Weicheng

#include <cmath>
#include <cstdlib>

#include "gtest/gtest.h"

class TestRiemannRotated : public ::testing::Test {
 protected:
  using Scalar = double;
  static Scalar rand_f() {
    return -1 + 2 * std::rand() / (1.0 + RAND_MAX);
  }
};

#include "mini/riemann/simple/single.hpp"
#include "mini/riemann/rotated/single.hpp"
class TestRiemannRotatedSingle : public TestRiemannRotated {
};
TEST_F(TestRiemannRotatedSingle, ThreeDimensional) {
  using Solver = mini::riemann::rotated::Single<Scalar, 3>;
  std::srand(31415926);
  using Value = typename Solver::Conservative;
  for (int i = 0; i < (1 << 5); ++i) {
    Scalar a_x = rand_f(), a_y = rand_f(), a_z = rand_f();
    Solver::SetJacobians(a_x, a_y, a_z);
    for (int j = 0; j < (1 << 5); ++j) {
      Value u = Value::Random();
      EXPECT_EQ(std::hypot(a_x, a_y, a_z), Solver::GetMaximumSpeed(u));
    }
  }
}

#include "mini/riemann/simple/burgers.hpp"
#include "mini/riemann/rotated/burgers.hpp"
class TestRiemannRotatedBurgers : public TestRiemannRotated {
};
TEST_F(TestRiemannRotatedBurgers, ThreeDimensional) {
  using Solver = mini::riemann::rotated::Burgers<Scalar, 3>;
  std::srand(31415926);
  using Value = typename Solver::Conservative;
  for (int i = 0; i < (1 << 5); ++i) {
    Scalar k_x = rand_f(), k_y = rand_f(), k_z = rand_f();
    Solver::SetJacobians(k_x, k_y, k_z);
    for (int j = 0; j < (1 << 5); ++j) {
      Value u = Value::Random();
      EXPECT_EQ(std::hypot(k_x, k_y, k_z) * std::abs(u[0]),
          Solver::GetMaximumSpeed(u));
    }
  }
}

#include "mini/riemann/simple/double.hpp"
#include "mini/riemann/rotated/double.hpp"
#include "mini/geometry/pi.hpp"
#include "mini/algebra/eigen.hpp"
class TestRiemannRotatedDouble : public TestRiemannRotated {
};
TEST_F(TestRiemannRotatedDouble, ThreeDimensional) {
  using Solver = mini::riemann::rotated::Double<Scalar, 3>;
  std::srand(31415926);
  using Value = typename Solver::Conservative;
  using Jacobian = typename Solver::Jacobian;
  for (int i = 0; i < (1 << 5); ++i) {
    // Get a diagonal system:
    Scalar lambda_0 = rand_f();
    EXPECT_GE(lambda_0, -1);
    EXPECT_LE(lambda_0, +1);
    Scalar lambda_1 = rand_f() + 100;
    EXPECT_GE(lambda_1, -1 + 100);
    EXPECT_LE(lambda_1, +1 + 100);
    Jacobian a_n = Jacobian{ {lambda_1, 0}, {0, lambda_0} };
    Value eigvals = mini::algebra::GetEigenValues(a_n);
    EXPECT_EQ(eigvals[0], lambda_0);
    EXPECT_EQ(eigvals[1], lambda_1);
    // Get an orientation:
    auto [cos_theta, sin_theta] = mini::geometry::CosSin(rand_f() + 180.0);
    auto [cos_phi, sin_phi] = mini::geometry::CosSin(rand_f() + 180.0);
    Scalar n_x = sin_theta * cos_phi;
    Scalar n_y = sin_theta * sin_phi;
    Scalar n_z = cos_theta;
    // Rotate the diagonal system by the orientation:
    Jacobian a_x = a_n * n_x;
    Jacobian a_y = a_n * n_y;
    Jacobian a_z = a_n * n_z;
    Solver::SetJacobians(a_x, a_y, a_z);
    for (int j = 0; j < (1 << 5); ++j) {
      Value u = Value::Random();
      EXPECT_NEAR(lambda_1, Solver::GetMaximumSpeed(u), 1.0);
    }
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
