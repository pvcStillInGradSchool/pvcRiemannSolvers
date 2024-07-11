//  Copyright 2023 PEI Weicheng

#include <iostream>
#include <cstdlib>

#include "mini/integrator/function.hpp"
#include "mini/integrator/tetrahedron.hpp"
#include "mini/coordinate/tetrahedron.hpp"
#include "mini/integrator/hexahedron.hpp"
#include "mini/coordinate/hexahedron.hpp"
#include "mini/integrator/triangle.hpp"
#include "mini/coordinate/triangle.hpp"
#include "mini/integrator/quadrangle.hpp"
#include "mini/coordinate/quadrangle.hpp"
#include "mini/basis/taylor.hpp"
#include "mini/rand.hpp"

#include "gtest/gtest.h"


class TestBasisTaylor : public ::testing::Test {
  void SetUp() override {
    std::srand(31415926);
  }
};
TEST_F(TestBasisTaylor, In1dSpace) {
  using Basis = mini::basis::Taylor<double, 1, 5>;
  static_assert(Basis::N == 6);
  double x = mini::rand::uniform(0., 1.);
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
  double x = mini::rand::uniform(0., 1.);
  double y = mini::rand::uniform(0., 1.);
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
  double x = mini::rand::uniform(0., 1.);
  double y = mini::rand::uniform(0., 1.);
  double z = mini::rand::uniform(0., 1.);
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
  using Coeff = typename Basis::Matrix<Basis::N, Basis::N>;
  Coeff pdv_expect = Coeff::Zero();
  pdv_expect(1, 1) = 1;  // (∂/∂x)(x)
  pdv_expect(2, 2) = 1;  // (∂/∂y)(y)
  pdv_expect(3, 3) = 1;  // (∂/∂z)(z)
  pdv_expect(4, 1) = 2*x;  //     (∂/∂x)(x*x)
  pdv_expect(4, 4) = 2;  // (∂/∂x)(∂/∂x)(x*x)
  pdv_expect(5, 1) = y;  //       (∂/∂x)(x*y)
  pdv_expect(5, 2) = x;  //       (∂/∂y)(x*y)
  pdv_expect(5, 5) = 1;  // (∂/∂x)(∂/∂y)(x*y)
  pdv_expect(6, 1) = z;  //       (∂/∂x)(x*z)
  pdv_expect(6, 3) = x;  //       (∂/∂z)(x*z)
  pdv_expect(6, 6) = 1;  // (∂/∂x)(∂/∂z)(x*z)
  pdv_expect(7, 2) = 2*y;  //     (∂/∂y)(y*y)
  pdv_expect(7, 7) = 2;  // (∂/∂y)(∂/∂y)(y*y)
  pdv_expect(8, 2) = z;  //       (∂/∂y)(y*z)
  pdv_expect(8, 3) = y;  //       (∂/∂z)(y*z)
  pdv_expect(8, 8) = 1;  // (∂/∂y)(∂/∂z)(y*z)
  pdv_expect(9, 3) = 2*z;  //     (∂/∂z)(z*z)
  pdv_expect(9, 9) = 2;  // (∂/∂z)(∂/∂z)(z*z)
  Coeff coeff = Coeff::Identity();
  auto pdv_actual = Basis::GetPartialDerivatives({ x, y, z }, coeff);
  EXPECT_EQ(pdv_expect, pdv_actual);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
