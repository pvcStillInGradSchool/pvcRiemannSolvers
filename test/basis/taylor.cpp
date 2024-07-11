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
  static_assert(Basis::CountBasis(0) == 1);
  static_assert(Basis::CountBasis(1) == 2);
  static_assert(Basis::CountBasis(2) == 3);
  static_assert(Basis::CountBasis(3) == 4);
  static_assert(Basis::CountBasis(4) == 5);
  static_assert(Basis::CountBasis(5) == 6);
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
  static_assert(Basis::CountBasis(0) == 1);
  static_assert(Basis::CountBasis(1) == 3);
  static_assert(Basis::CountBasis(2) == 6);
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
  static_assert(Basis::CountBasis(0) == 1);
  static_assert(Basis::CountBasis(1) == 4);
  static_assert(Basis::CountBasis(2) == 10);
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
TEST_F(TestBasisTaylor, D3P3) {
  using Basis = mini::basis::Taylor<double, 3, 3>;
  using Index = typename Basis::Index;
  static_assert(Basis::CountBasis(0) == 1);
  static_assert(Basis::CountBasis(1) == 4);
  static_assert(Basis::CountBasis(2) == 10);
  static_assert(Basis::CountBasis(3) == 20);
  static_assert(Basis::N == 20);
  double x = mini::rand::uniform(1., 2.);
  double y = mini::rand::uniform(1., 2.);
  double z = mini::rand::uniform(1., 2.);
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
  EXPECT_EQ(res[Index::XXX], x * x * x);
  EXPECT_NEAR(res[Index::XXY], x * x * y, 1e-15);
  EXPECT_NEAR(res[Index::XXZ], x * x * z, 1e-15);
  EXPECT_EQ(res[Index::XYY], x * y * y);
  EXPECT_NEAR(res[Index::XYZ], x * y * z, 1e-15);
  EXPECT_EQ(res[Index::XZZ], x * z * z);
  EXPECT_EQ(res[Index::YYY], y * y * y);
  EXPECT_EQ(res[Index::YYZ], y * y * z);
  EXPECT_EQ(res[Index::YZZ], y * z * z);
  EXPECT_EQ(res[Index::ZZZ], z * z * z);
  using Coeff = typename Basis::Matrix<Basis::N, Basis::N>;
  Coeff coeff = Coeff::Identity();
  auto pdv = Basis::GetPartialDerivatives({ x, y, z }, coeff);
  // pdv_expect[func_index][pdv_index]
  // minus non-zero pdvs of x, y, z
  EXPECT_EQ(0, pdv(Index::X, Index::X) -= 1);
  EXPECT_EQ(0, pdv(Index::Y, Index::Y) -= 1);
  EXPECT_EQ(0, pdv(Index::Z, Index::Z) -= 1);
  // minus non-zero pdvs of xx
  EXPECT_EQ(0, pdv(Index::XX, Index::X) -= 2 * x);
  EXPECT_EQ(0, pdv(Index::XX, Index::XX) -= 2);
  // minus non-zero pdvs of xy
  EXPECT_EQ(0, pdv(Index::XY, Index::X) -= y);
  EXPECT_EQ(0, pdv(Index::XY, Index::Y) -= x);
  EXPECT_EQ(0, pdv(Index::XY, Index::XY) -= 1);
  // minus non-zero pdvs of xz
  EXPECT_EQ(0, pdv(Index::XZ, Index::X) -= z);
  EXPECT_EQ(0, pdv(Index::XZ, Index::Z) -= x);
  EXPECT_EQ(0, pdv(Index::XZ, Index::XZ) -= 1);
  // minus non-zero pdvs of yy
  EXPECT_EQ(0, pdv(Index::YY, Index::Y) -= 2 * y);
  EXPECT_EQ(0, pdv(Index::YY, Index::YY) -= 2);
  // minus non-zero pdvs of yz
  EXPECT_EQ(0, pdv(Index::YZ, Index::Y) -= z);
  EXPECT_EQ(0, pdv(Index::YZ, Index::Z) -= y);
  EXPECT_EQ(0, pdv(Index::YZ, Index::YZ) -= 1);
  // minus non-zero pdvs of zz
  EXPECT_EQ(0, pdv(Index::ZZ, Index::Z) -= 2 * z);
  EXPECT_EQ(0, pdv(Index::ZZ, Index::ZZ) -= 2);
  // minus non-zero pdvs of xxx
  EXPECT_EQ(0, pdv(Index::XXX, Index::X) -= 3 * (x * x));
  EXPECT_EQ(0, pdv(Index::XXX, Index::XX) -= 3 * 2 * x);
  EXPECT_EQ(0, pdv(Index::XXX, Index::XXX) -= 3 * 2 * 1);
  // minus non-zero pdvs of xxy
  EXPECT_EQ(0, pdv(Index::XXY, Index::X) -= 2 * x * y);
  EXPECT_EQ(0, pdv(Index::XXY, Index::Y) -= x * x);
  EXPECT_EQ(0, pdv(Index::XXY, Index::XX) -= 2 * y);
  EXPECT_EQ(0, pdv(Index::XXY, Index::XY) -= 2 * x);
  EXPECT_EQ(0, pdv(Index::XXY, Index::XXY) -= 2);
  // minus non-zero pdvs of xxz
  EXPECT_EQ(0, pdv(Index::XXZ, Index::X) -= 2 * x * z);
  EXPECT_EQ(0, pdv(Index::XXZ, Index::Z) -= x * x);
  EXPECT_EQ(0, pdv(Index::XXZ, Index::XX) -= 2 * z);
  EXPECT_EQ(0, pdv(Index::XXZ, Index::XZ) -= 2 * x);
  EXPECT_EQ(0, pdv(Index::XXZ, Index::XXZ) -= 2);
  // minus non-zero pdvs of xyy
  EXPECT_EQ(0, pdv(Index::XYY, Index::X) -= y * y);
  EXPECT_EQ(0, pdv(Index::XYY, Index::Y) -= 2 * x * y);
  EXPECT_EQ(0, pdv(Index::XYY, Index::XY) -= 2 * y);
  EXPECT_EQ(0, pdv(Index::XYY, Index::YY) -= 2 * x);
  EXPECT_EQ(0, pdv(Index::XYY, Index::XYY) -= 2);
  // minus non-zero pdvs of xyz
  EXPECT_EQ(0, pdv(Index::XYZ, Index::X) -= y * z);
  EXPECT_EQ(0, pdv(Index::XYZ, Index::Y) -= x * z);
  EXPECT_EQ(0, pdv(Index::XYZ, Index::Z) -= x * y);
  EXPECT_EQ(0, pdv(Index::XYZ, Index::XY) -= z);
  EXPECT_EQ(0, pdv(Index::XYZ, Index::XZ) -= y);
  EXPECT_EQ(0, pdv(Index::XYZ, Index::YZ) -= x);
  EXPECT_EQ(0, pdv(Index::XYZ, Index::XYZ) -= 1);
  // minus non-zero pdvs of xzz
  EXPECT_EQ(0, pdv(Index::XZZ, Index::X) -= z * z);
  EXPECT_EQ(0, pdv(Index::XZZ, Index::Z) -= 2 * x * z);
  EXPECT_EQ(0, pdv(Index::XZZ, Index::XZ) -= 2 * z);
  EXPECT_EQ(0, pdv(Index::XZZ, Index::ZZ) -= 2 * x);
  EXPECT_EQ(0, pdv(Index::XZZ, Index::XZZ) -= 2);
  // minus non-zero pdvs of yyy
  EXPECT_EQ(0, pdv(Index::YYY, Index::Y) -= 3 * (y * y));
  EXPECT_EQ(0, pdv(Index::YYY, Index::YY) -= 3 * 2 * y);
  EXPECT_EQ(0, pdv(Index::YYY, Index::YYY) -= 3 * 2 * 1);
  // minus non-zero pdvs of yyz
  EXPECT_EQ(0, pdv(Index::YYZ, Index::Y) -= 2 * y * z);
  EXPECT_EQ(0, pdv(Index::YYZ, Index::Z) -= y * y);
  EXPECT_EQ(0, pdv(Index::YYZ, Index::YY) -= 2 * z);
  EXPECT_EQ(0, pdv(Index::YYZ, Index::YZ) -= 2 * y);
  EXPECT_EQ(0, pdv(Index::YYZ, Index::YYZ) -= 2);
  // minus non-zero pdvs of yzz
  EXPECT_EQ(0, pdv(Index::YZZ, Index::Y) -= z * z);
  EXPECT_EQ(0, pdv(Index::YZZ, Index::Z) -= 2 * y * z);
  EXPECT_EQ(0, pdv(Index::YZZ, Index::YZ) -= 2 * z);
  EXPECT_EQ(0, pdv(Index::YZZ, Index::ZZ) -= 2 * y);
  EXPECT_EQ(0, pdv(Index::YZZ, Index::YZZ) -= 2);
  // minus non-zero pdvs of zzz
  EXPECT_EQ(0, pdv(Index::ZZZ, Index::Z) -= 3 * (z * z));
  EXPECT_EQ(0, pdv(Index::ZZZ, Index::ZZ) -= 3 * 2 * z);
  EXPECT_EQ(0, pdv(Index::ZZZ, Index::ZZZ) -= 3 * 2 * 1);
  // assert all zeros
  EXPECT_NEAR(pdv.norm(), 0, 1e-14);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
