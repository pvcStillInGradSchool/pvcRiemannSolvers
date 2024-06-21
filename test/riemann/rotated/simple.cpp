// Copyright 2024 PEI Weicheng

#include <cmath>
#include <cstdlib>

#include "gtest/gtest.h"

#include "mini/riemann/simple/burgers.hpp"
#include "mini/riemann/rotated/burgers.hpp"

class TestRiemannRotatedBurgers : public ::testing::Test {
 protected:
  using Scalar = double;
  static Scalar rand_f() {
    return -1 + 2 * std::rand() / (1.0 + RAND_MAX);
  }
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

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
