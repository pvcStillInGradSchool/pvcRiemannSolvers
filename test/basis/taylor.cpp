//  Copyright 2023 PEI Weicheng

#include <iostream>
#include <cstdlib>

#include "mini/gauss/function.hpp"
#include "mini/gauss/tetrahedron.hpp"
#include "mini/coordinate/tetrahedron.hpp"
#include "mini/gauss/hexahedron.hpp"
#include "mini/coordinate/hexahedron.hpp"
#include "mini/gauss/triangle.hpp"
#include "mini/coordinate/triangle.hpp"
#include "mini/gauss/quadrangle.hpp"
#include "mini/coordinate/quadrangle.hpp"
#include "mini/basis/taylor.hpp"

#include "gtest/gtest.h"

double rand_f() {
  return std::rand() / (1.0 + RAND_MAX);
}

class TestBasisTaylor : public ::testing::Test {
};
TEST_F(TestBasisTaylor, In1dSpace) {
  using Basis = mini::basis::Taylor<double, 1, 5>;
  static_assert(Basis::N == 6);
  std::srand(31415926);
  double x{rand_f()};
  EXPECT_NE(x, 0);
  typename Basis::Vector res;
  res = Basis::GetValues(x);
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[1], x);
  EXPECT_EQ(res[2], x * x);
  EXPECT_EQ(res[3], x * x * x);
  EXPECT_EQ(res[4], x * x * x * x);
  EXPECT_EQ(res[5], x * x * x * x * x);
  res = Basis::GetDerivatives(0, x);
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[1], x);
  EXPECT_EQ(res[2], x * x);
  EXPECT_EQ(res[3], x * x * x);
  EXPECT_NEAR(res[4], x * x * x * x, 1e-16);
  EXPECT_NEAR(res[5], x * x * x * x * x, 1e-16);
  res = Basis::GetDerivatives(1, x);
  EXPECT_EQ(res[0], 0);
  EXPECT_EQ(res[1], 1);
  EXPECT_EQ(res[2], 2 * x);
  EXPECT_EQ(res[3], 3 * x * x);
  EXPECT_EQ(res[4], 4 * x * x * x);
  EXPECT_EQ(res[5], 5 * x * x * x * x);
  res = Basis::GetDerivatives(2, x);
  EXPECT_EQ(res[0], 0);
  EXPECT_EQ(res[1], 0);
  EXPECT_EQ(res[2], 2 * 1);
  EXPECT_EQ(res[3], 3 * 2 * x);
  EXPECT_EQ(res[4], 4 * 3 * x * x);
  EXPECT_EQ(res[5], 5 * 4 * x * x * x);
  res = Basis::GetDerivatives(3, x);
  EXPECT_EQ(res[0], 0);
  EXPECT_EQ(res[1], 0);
  EXPECT_EQ(res[2], 0);
  EXPECT_EQ(res[3], 3 * 2 * 1);
  EXPECT_EQ(res[4], 4 * 3 * 2 * x);
  EXPECT_EQ(res[5], 5 * 4 * 3 * x * x);
  res = Basis::GetDerivatives(4, x);
  EXPECT_EQ(res[0], 0);
  EXPECT_EQ(res[1], 0);
  EXPECT_EQ(res[2], 0);
  EXPECT_EQ(res[3], 0);
  EXPECT_EQ(res[4], 4 * 3 * 2 * 1);
  EXPECT_EQ(res[5], 5 * 4 * 3 * 2 * x);
  res = Basis::GetDerivatives(5, x);
  EXPECT_EQ(res[0], 0);
  EXPECT_EQ(res[1], 0);
  EXPECT_EQ(res[2], 0);
  EXPECT_EQ(res[3], 0);
  EXPECT_EQ(res[4], 0);
  EXPECT_EQ(res[5], 5 * 4 * 3 * 2 * 1);
}
TEST_F(TestBasisTaylor, In2dSpace) {
  using Basis = mini::basis::Taylor<double, 2, 2>;
  static_assert(Basis::N == 6);
  std::srand(31415926);
  double x{rand_f()}, y{rand_f()};
  typename Basis::MatNx1 res;
  res = Basis::GetValue({x, y});
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[1], x);
  EXPECT_EQ(res[2], y);
  EXPECT_EQ(res[3], x * x);
  EXPECT_EQ(res[4], x * y);
  EXPECT_EQ(res[5], y * y);
  x = 0.3; y = 0.4;
  res = Basis::GetValue({x, y});
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[1], x);
  EXPECT_EQ(res[2], y);
  EXPECT_EQ(res[3], x * x);
  EXPECT_EQ(res[4], x * y);
  EXPECT_EQ(res[5], y * y);
}
TEST_F(TestBasisTaylor, In3dSpace) {
  using Basis = mini::basis::Taylor<double, 3, 2>;
  static_assert(Basis::N == 10);
  std::srand(31415926);
  double x{rand_f()}, y{rand_f()}, z{rand_f()};
  typename Basis::MatNx1 res;
  res = Basis::GetValue({x, y, z});
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[1], x);
  EXPECT_EQ(res[2], y);
  EXPECT_EQ(res[3], z);
  EXPECT_EQ(res[4], x * x);
  EXPECT_EQ(res[5], x * y);
  EXPECT_EQ(res[6], x * z);
  EXPECT_EQ(res[7], y * y);
  EXPECT_EQ(res[8], y * z);
  EXPECT_EQ(res[9], z * z);
  x = 0.3; y = 0.4, z = 0.5;
  res = Basis::GetValue({x, y, z});
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[1], x);
  EXPECT_EQ(res[2], y);
  EXPECT_EQ(res[3], z);
  EXPECT_EQ(res[4], x * x);
  EXPECT_EQ(res[5], x * y);
  EXPECT_EQ(res[6], x * z);
  EXPECT_EQ(res[7], y * y);
  EXPECT_EQ(res[8], y * z);
  EXPECT_EQ(res[9], z * z);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
