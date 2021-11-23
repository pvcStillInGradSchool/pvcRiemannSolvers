//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/integrator/function.hpp"
#include "mini/integrator/hexa.hpp"
#include "mini/polynomial/basis.hpp"
#include "mini/polynomial/projection.hpp"

#include "gtest/gtest.h"

class TestProjection : public ::testing::Test {
 protected:
  using Raw = mini::polynomial::Raw<double, 3, 2>;
  using Basis = mini::polynomial::OrthoNormal<double, 3, 2>;
  using Gauss = mini::integrator::Hexa<double, 4, 4, 4>;
  using Coord = typename Gauss::GlobalCoord;
  Gauss gauss_;
};
TEST_F(TestProjection, ScalarFunction) {
  auto func = [](Coord const &point){
    auto x = point[0], y = point[1], z = point[2];
    return x * x + y * y + z * z;
  };
  using ProjFunc = mini::polynomial::Projection<double, 3, 2, 1>;
  auto basis = Basis(gauss_);
  auto projection = ProjFunc(func, basis);
  static_assert(ProjFunc::K == 1);
  static_assert(ProjFunc::N == 10);
  EXPECT_NEAR(projection({0, 0, 0})[0], 0.0, 1e-15);
  EXPECT_DOUBLE_EQ(projection({0.3, 0.4, 0.5})[0], 0.5);
  auto integral_f = mini::integrator::Integrate(func, gauss_);
  auto integral_1 = mini::integrator::Integrate([](auto const &){
    return 1.0;
  }, gauss_);
  EXPECT_NEAR(projection.GetAverage()[0], integral_f / integral_1, 1e-14);
}
TEST_F(TestProjection, VectorFunction) {
  using ProjFunc = mini::polynomial::Projection<double, 3, 2, 10>;
  using MatKx1 = typename ProjFunc::MatKx1;
  auto func = [](Coord const &point){
    auto x = point[0], y = point[1], z = point[2];
    MatKx1 res = { 1, x, y, z, x * x, x * y, x * z, y * y, y * z, z * z };
    return res;
  };
  auto basis = Basis(gauss_);
  auto projection = ProjFunc(func, basis);
  static_assert(ProjFunc::K == 10);
  static_assert(ProjFunc::N == 10);
  auto v_actual = projection({0.3, 0.4, 0.5});
  auto v_expect = Raw::CallAt({0.3, 0.4, 0.5});
  MatKx1 res = v_actual - v_expect;
  EXPECT_NEAR(v_actual[0], v_expect[0], 1e-14);
  EXPECT_NEAR(v_actual[1], v_expect[1], 1e-15);
  EXPECT_NEAR(v_actual[2], v_expect[2], 1e-15);
  EXPECT_DOUBLE_EQ(v_actual[3], v_expect[3]);
  EXPECT_NEAR(v_actual[4], v_expect[4], 1e-16);
  EXPECT_NEAR(v_actual[5], v_expect[5], 1e-16);
  EXPECT_DOUBLE_EQ(v_actual[6], v_expect[6]);
  EXPECT_NEAR(v_actual[7], v_expect[7], 1e-15);
  EXPECT_NEAR(v_actual[8], v_expect[8], 1e-16);
  EXPECT_NEAR(v_actual[9], v_expect[9], 1e-15);
  auto integral_f = mini::integrator::Integrate(func, gauss_);
  auto integral_1 = mini::integrator::Integrate([](auto const &){
    return 1.0;
  }, gauss_);
  res = projection.GetAverage() - integral_f / integral_1;
  EXPECT_NEAR(res.cwiseAbs().maxCoeff(), 0.0, 1e-14);
}
TEST_F(TestProjection, CoeffConsistency) {
  using ProjFunc = mini::polynomial::Projection<double, 3, 2, 5>;
  using MatKx1 = typename ProjFunc::MatKx1;
  auto func = [](Coord const &point){
    auto x = point[0], y = point[1], z = point[2];
    MatKx1 res = { std::sin(x + y), std::cos(y + z), std::tan(x * z),
        std::exp(y * z), std::log(z * z) };
    return res;
  };
  auto basis = Basis(gauss_);
  auto projection = ProjFunc(func, basis);
  auto coeff_diff = projection.GetCoeffOnRawBasis()
      - projection.GetCoeffOnOrthoNormalBasis() * basis.coeff();
  std::cout << projection.GetCoeffOnRawBasis() << std::endl;
  EXPECT_NEAR(coeff_diff.cwiseAbs().maxCoeff(), 0.0, 1e-14);
}
TEST_F(TestProjection, PartialDerivatives) {
  using ProjFunc = mini::polynomial::Projection<double, 3, 2, 10>;
  using Raw = mini::polynomial::Raw<double, 3, 2>;
  using MatKx1 = typename ProjFunc::MatKx1;
  auto func = [](Coord const &point) {
    return Raw::CallAt(point);
  };
  auto basis = Basis(gauss_);
  auto projection = ProjFunc(func, basis);
  static_assert(ProjFunc::K == 10);
  static_assert(ProjFunc::N == 10);
  auto point = Coord{ 0.3, 0.4, 0.5 };
  auto pdv_actual = projection.GetPdvValue(point);
  auto coeff = ProjFunc::MatKxN(); coeff.setIdentity();
  auto pdv_expect = Raw::GetPdvValue(point, coeff);
  ProjFunc::MatKxN diff = pdv_actual - pdv_expect;
  EXPECT_NEAR(diff.cwiseAbs().maxCoeff(), 0.0, 1e-14);
  auto s_actual = projection.GetSmoothness();
  std::cout << "s_actual.transpose() =\n" << s_actual.transpose() << std::endl;
  EXPECT_NEAR(s_actual[0], 0.0, 1e-14);
  EXPECT_NEAR(s_actual[1], 8.0, 1e-13);
  EXPECT_NEAR(s_actual[2], 8.0, 1e-14);
  EXPECT_NEAR(s_actual[3], 8.0, 1e-14);
  EXPECT_NEAR(s_actual[4], 80.0/3, 1e-13);
  EXPECT_NEAR(s_actual[5], 64.0/3, 1e-13);
  EXPECT_NEAR(s_actual[6], 64.0/3, 1e-13);
  EXPECT_NEAR(s_actual[7], 80.0/3, 1e-12);
  EXPECT_NEAR(s_actual[8], 64.0/3, 1e-13);
  EXPECT_NEAR(s_actual[9], 80.0/3, 1e-12);
}
TEST_F(TestProjection, ArithmeticOperators) {
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}