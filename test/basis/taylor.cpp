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
TEST_F(TestBasisTaylor, D1P5) {
  using Basis = mini::basis::Taylor<double, 1, 5>;
  using Index = typename Basis::Index;
  static_assert(Basis::N == 6);
  double x = mini::rand::uniform(0., 1.);
  EXPECT_NE(x, 0);
  typename Basis::Vector res;
  res = Basis::GetValues(x);
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[Index::X], x);
  EXPECT_EQ(res[Index::XX], x * x);
  EXPECT_EQ(res[Index::XXX], x * x * x);
  EXPECT_EQ(res[Index::XXXX], x * x * x * x);
  EXPECT_EQ(res[Index::XXXXX], x * x * x * x * x);
  res = Basis::GetDerivatives(0, x);
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[Index::X], x);
  EXPECT_EQ(res[Index::XX], x * x);
  EXPECT_EQ(res[Index::XXX], x * x * x);
  EXPECT_NEAR(res[Index::XXXX], x * x * x * x, 1e-16);
  EXPECT_NEAR(res[Index::XXXXX], x * x * x * x * x, 1e-16);
  res = Basis::GetDerivatives(1, x);
  EXPECT_EQ(res[0], 0);
  EXPECT_EQ(res[Index::X], 1);
  EXPECT_EQ(res[Index::XX], 2 * x);
  EXPECT_EQ(res[Index::XXX], 3 * x * x);
  EXPECT_EQ(res[Index::XXXX], 4 * x * x * x);
  EXPECT_EQ(res[Index::XXXXX], 5 * x * x * x * x);
  res = Basis::GetDerivatives(2, x);
  EXPECT_EQ(res[0], 0);
  EXPECT_EQ(res[Index::X], 0);
  EXPECT_EQ(res[Index::XX], 2 * 1);
  EXPECT_EQ(res[Index::XXX], 3 * 2 * x);
  EXPECT_EQ(res[Index::XXXX], 4 * 3 * x * x);
  EXPECT_EQ(res[Index::XXXXX], 5 * 4 * x * x * x);
  res = Basis::GetDerivatives(3, x);
  EXPECT_EQ(res[0], 0);
  EXPECT_EQ(res[Index::X], 0);
  EXPECT_EQ(res[Index::XX], 0);
  EXPECT_EQ(res[Index::XXX], 3 * 2 * 1);
  EXPECT_EQ(res[Index::XXXX], 4 * 3 * 2 * x);
  EXPECT_EQ(res[Index::XXXXX], 5 * 4 * 3 * x * x);
  res = Basis::GetDerivatives(4, x);
  EXPECT_EQ(res[0], 0);
  EXPECT_EQ(res[Index::X], 0);
  EXPECT_EQ(res[Index::XX], 0);
  EXPECT_EQ(res[Index::XXX], 0);
  EXPECT_EQ(res[Index::XXXX], 4 * 3 * 2 * 1);
  EXPECT_EQ(res[Index::XXXXX], 5 * 4 * 3 * 2 * x);
  res = Basis::GetDerivatives(5, x);
  EXPECT_EQ(res[0], 0);
  EXPECT_EQ(res[Index::X], 0);
  EXPECT_EQ(res[Index::XX], 0);
  EXPECT_EQ(res[Index::XXX], 0);
  EXPECT_EQ(res[Index::XXXX], 0);
  EXPECT_EQ(res[Index::XXXXX], 5 * 4 * 3 * 2 * 1);
}
TEST_F(TestBasisTaylor, D2P2) {
  using Basis = mini::basis::Taylor<double, 2, 2>;
  using Index = typename Basis::Index;
  static_assert(Basis::N == 6);
  double x = mini::rand::uniform(0., 1.);
  double y = mini::rand::uniform(0., 1.);
  typename Basis::MatNx1 res;
  res = Basis::GetValue({x, y});
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[Index::X], x);
  EXPECT_EQ(res[Index::Y], y);
  EXPECT_EQ(res[Index::XX], x * x);
  EXPECT_EQ(res[Index::XY], x * y);
  EXPECT_EQ(res[Index::YY], y * y);
  x = 0.3; y = 0.4;
  res = Basis::GetValue({x, y});
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[Index::X], x);
  EXPECT_EQ(res[Index::Y], y);
  EXPECT_EQ(res[Index::XX], x * x);
  EXPECT_EQ(res[Index::XY], x * y);
  EXPECT_EQ(res[Index::YY], y * y);
}
TEST_F(TestBasisTaylor, D3P2) {
  using Basis = mini::basis::Taylor<double, 3, 2>;
  using Index = typename Basis::Index;
  static_assert(Basis::N == 10);
  double x = mini::rand::uniform(0., 1.);
  double y = mini::rand::uniform(0., 1.);
  double z = mini::rand::uniform(0., 1.);
  typename Basis::MatNx1 res;
  res = Basis::GetValue({x, y, z});
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[Index::X], x);
  EXPECT_EQ(res[Index::Y], y);
  EXPECT_EQ(res[Index::Z], z);
  EXPECT_EQ(res[Index::XX], x * x);
  EXPECT_EQ(res[Index::XY], x * y);
  EXPECT_EQ(res[Index::XZ], x * z);
  EXPECT_EQ(res[Index::YY], y * y);
  EXPECT_EQ(res[Index::YZ], y * z);
  EXPECT_EQ(res[Index::ZZ], z * z);
  x = 0.3; y = 0.4, z = 0.5;
  res = Basis::GetValue({x, y, z});
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[Index::X], x);
  EXPECT_EQ(res[Index::Y], y);
  EXPECT_EQ(res[Index::Z], z);
  EXPECT_EQ(res[Index::XX], x * x);
  EXPECT_EQ(res[Index::XY], x * y);
  EXPECT_EQ(res[Index::XZ], x * z);
  EXPECT_EQ(res[Index::YY], y * y);
  EXPECT_EQ(res[Index::YZ], y * z);
  EXPECT_EQ(res[Index::ZZ], z * z);
  using Coeff = typename Basis::Matrix<Basis::N, Basis::N>;
  Coeff pdv_expect = Coeff::Zero();
  // pdv_expect[func_index][pdv_index]
  pdv_expect(Index::X, Index::X) = 1;
  pdv_expect(Index::Y, Index::Y) = 1;
  pdv_expect(Index::Z, Index::Z) = 1;
  pdv_expect(Index::XX, Index::X) = 2 * x;
  pdv_expect(Index::XX, Index::XX) = 2;
  pdv_expect(Index::XY, Index::X) = y;
  pdv_expect(Index::XY, Index::Y) = x;
  pdv_expect(Index::XY, Index::XY) = 1;
  pdv_expect(Index::XZ, Index::X) = z;
  pdv_expect(Index::XZ, Index::Z) = x;
  pdv_expect(Index::XZ, Index::XZ) = 1;
  pdv_expect(Index::YY, Index::Y) = 2 * y;
  pdv_expect(Index::YY, Index::YY) = 2;
  pdv_expect(Index::YZ, Index::Y) = z;
  pdv_expect(Index::YZ, Index::Z) = y;
  pdv_expect(Index::YZ, Index::YZ) = 1;
  pdv_expect(Index::ZZ, Index::Z) = 2 * z;
  pdv_expect(Index::ZZ, Index::ZZ) = 2;
  Coeff coeff = Coeff::Identity();
  auto pdv_actual = Basis::GetPartialDerivatives({ x, y, z }, coeff);
  EXPECT_EQ(pdv_expect, pdv_actual);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
