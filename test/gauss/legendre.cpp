// Copyright 2019 PEI Weicheng and YANG Minghao

#include "mini/gauss/legendre.hpp"

#include "gtest/gtest.h"

class TestLegendre : public ::testing::Test {
 protected:
  using Scalar = double;
};
TEST_F(TestLegendre, OnePoint) {
  using Gauss = mini::gauss::Legendre<Scalar, 1>;
  auto &p = Gauss::points;
  auto &w = Gauss::weights;
  EXPECT_DOUBLE_EQ(p[0], 0.0);
  EXPECT_DOUBLE_EQ(w[0], 2.0);
}
TEST_F(TestLegendre, TwoPoint) {
  using Gauss = mini::gauss::Legendre<Scalar, 2>;
  auto &p = Gauss::points;
  auto &w = Gauss::weights;
  EXPECT_DOUBLE_EQ(p[0], -0.5773502691896257);
  EXPECT_DOUBLE_EQ(w[0], 1.0);
  EXPECT_DOUBLE_EQ(p[1], +0.5773502691896257);
  EXPECT_DOUBLE_EQ(w[1], 1.0);
}
TEST_F(TestLegendre, ThreePoint) {
  using Gauss = mini::gauss::Legendre<Scalar, 3>;
  auto &p = Gauss::points;
  auto &w = Gauss::weights;
  EXPECT_DOUBLE_EQ(p[0], -0.7745966692414834);
  EXPECT_DOUBLE_EQ(w[0], 0.5555555555555556);
  EXPECT_DOUBLE_EQ(p[1], 0.0);
  EXPECT_DOUBLE_EQ(w[1], 0.8888888888888888);
  EXPECT_DOUBLE_EQ(p[2], +0.7745966692414834);
  EXPECT_DOUBLE_EQ(w[2], 0.5555555555555556);
}
TEST_F(TestLegendre, FourPoint) {
  using Gauss = mini::gauss::Legendre<Scalar, 4>;
  auto &p = Gauss::points;
  auto &w = Gauss::weights;
  EXPECT_DOUBLE_EQ(p[0], -0.8611363115940526);
  EXPECT_DOUBLE_EQ(w[0], 0.34785484513745385);
  EXPECT_DOUBLE_EQ(p[1], -0.3399810435848563);
  EXPECT_DOUBLE_EQ(w[1], 0.6521451548625462);
  EXPECT_DOUBLE_EQ(p[2], +0.3399810435848563);
  EXPECT_DOUBLE_EQ(w[2], 0.6521451548625462);
  EXPECT_DOUBLE_EQ(p[3], +0.8611363115940526);
  EXPECT_DOUBLE_EQ(w[3], 0.34785484513745385);
}
TEST_F(TestLegendre, FivePoint) {
  using Gauss = mini::gauss::Legendre<Scalar, 5>;
  auto &p = Gauss::points;
  auto &w = Gauss::weights;
  EXPECT_DOUBLE_EQ(p[0], -0.906179845938664);
  EXPECT_DOUBLE_EQ(w[0], 0.23692688505618908);
  EXPECT_DOUBLE_EQ(p[1], -0.5384693101056831);
  EXPECT_DOUBLE_EQ(w[1], 0.47862867049936647);
  EXPECT_DOUBLE_EQ(p[2], 0.0);
  EXPECT_DOUBLE_EQ(w[2], 0.5688888888888889);
  EXPECT_DOUBLE_EQ(p[3], +0.5384693101056831);
  EXPECT_DOUBLE_EQ(w[3], 0.47862867049936647);
  EXPECT_DOUBLE_EQ(p[4], +0.906179845938664);
  EXPECT_DOUBLE_EQ(w[4], 0.23692688505618908);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}