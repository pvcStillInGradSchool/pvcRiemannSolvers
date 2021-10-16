//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/integrator/function.hpp"
#include "mini/integrator/hexa.hpp"

#include "gtest/gtest.h"

using std::sqrt;

namespace mini {
namespace integrator {

class TestHexa4x4x4 : public ::testing::Test {
 protected:
  using Hexa4x4x4 = Hexa<double, 4, 4, 4>;
  using Mat1x8 = algebra::Matrix<double, 1, 8>;
  using Mat3x8 = algebra::Matrix<double, 3, 8>;
  using Mat3x1 = algebra::Matrix<double, 3, 1>;
  using Basis = OrthoNormalBasis<double, 3, 2>;
  using Y = typename Basis::MatNx1;
  using A = typename Basis::MatNxN;
  using ScalarPF = Projection<double, 3, 2, 1>;
  using Mat1x10 = algebra::Matrix<double, 1, 10>;
  using VectorPF = Projection<double, 3, 2, 11>;
  using Mat11x1 = algebra::Matrix<double, 11, 1>;
  using Mat11x10 = algebra::Matrix<double, 11, 10>;
};
TEST_F(TestHexa4x4x4, VirtualMethods) {
  Mat3x8 xyz_global_i;
  xyz_global_i.row(0) << -1, +1, +1, -1, -1, +1, +1, -1;
  xyz_global_i.row(1) << -1, -1, +1, +1, -1, -1, +1, +1;
  xyz_global_i.row(2) << -1, -1, -1, -1, +1, +1, +1, +1;
  auto hexa = Hexa4x4x4(xyz_global_i);
  static_assert(hexa.CellDim() == 3);
  static_assert(hexa.PhysDim() == 3);
  EXPECT_EQ(hexa.CountQuadPoints(), 64);
  auto p0 = hexa.GetLocalCoord(0);
  EXPECT_EQ(p0[0], -std::sqrt((3 - 2 * std::sqrt(1.2)) / 7));
  EXPECT_EQ(p0[1], -std::sqrt((3 - 2 * std::sqrt(1.2)) / 7));
  EXPECT_EQ(p0[2], -std::sqrt((3 - 2 * std::sqrt(1.2)) / 7));
  auto w1d = (18 + std::sqrt(30)) / 36.0;
  EXPECT_EQ(hexa.GetLocalWeight(0), w1d * w1d * w1d);
}
TEST_F(TestHexa4x4x4, CommonMethods) {
  Mat3x8 xyz_global_i;
  xyz_global_i.row(0) << -1, +1, +1, -1, -1, +1, +1, -1;
  xyz_global_i.row(1) << -1, -1, +1, +1, -1, -1, +1, +1;
  xyz_global_i.row(2) << -1, -1, -1, -1, +1, +1, +1, +1;
  auto hexa = Hexa4x4x4(xyz_global_i);
  EXPECT_EQ(hexa.LocalToGlobal(1, 1, 1), Mat3x1(1, 1, 1));
  EXPECT_EQ(hexa.LocalToGlobal(1.5, 1.5, 1.5), Mat3x1(1.5, 1.5, 1.5));
  EXPECT_EQ(hexa.LocalToGlobal(3, 4, 5), Mat3x1(3, 4, 5));
  EXPECT_EQ(hexa.global_to_local_3x1(3, 4, 2), Mat3x1(3, 4, 2));
  EXPECT_EQ(hexa.global_to_local_3x1(4, 5.5, 2.5), Mat3x1(4, 5.5, 2.5));
  EXPECT_EQ(hexa.global_to_local_3x1(7, 13, 6), Mat3x1(7, 13, 6));
  EXPECT_DOUBLE_EQ(Quadrature([](Mat3x1 const&){ return 2.0; }, hexa), 16.0);
  EXPECT_DOUBLE_EQ(Integrate([](Mat3x1 const&){ return 2.0; }, hexa), 16.0);
  auto f = [](Mat3x1 const& xyz){ return xyz[0]; };
  auto g = [](Mat3x1 const& xyz){ return xyz[1]; };
  auto h = [](Mat3x1 const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, hexa), Integrate(h, hexa));
  EXPECT_DOUBLE_EQ(Norm(f, hexa), sqrt(Innerprod(f, f, hexa)));
  EXPECT_DOUBLE_EQ(Norm(g, hexa), sqrt(Innerprod(g, g, hexa)));
}
TEST_F(TestHexa4x4x4, OrthoNormalBasis) {
  // build a hexa-integrator
  Mat3x8 xyz_global_i;
  xyz_global_i.row(0) << -1, +1, +1, -1, -1, +1, +1, -1;
  xyz_global_i.row(1) << -1, -1, +1, +1, -1, -1, +1, +1;
  xyz_global_i.row(2) << -1, -1, -1, -1, +1, +1, +1, +1;
  auto hexa = Hexa4x4x4(xyz_global_i);
  // build an orthonormal basis on it
  auto basis = Basis(hexa);
  // check orthonormality
  double residual = (Integrate([&basis](const Mat3x1& xyz) {
    auto col = basis(xyz);
    A prod = col * col.transpose();
    return prod;
  }, hexa) - A::Identity()).cwiseAbs().maxCoeff();
  EXPECT_NEAR(residual, 0.0, 1e-14);
  // build another hexa-integrator
  Mat3x1 left = {-1, 2, 3};
  auto x = left[0], y = left[1], z = left[2];
  xyz_global_i.row(0) << x-1, x+1, x+1, x-1, x-1, x+1, x+1, x-1;
  xyz_global_i.row(1) << y-1, y-1, y+1, y+1, y-1, y-1, y+1, y+1;
  xyz_global_i.row(2) << z-1, z-1, z-1, z-1, z+1, z+1, z+1, z+1;
  hexa = Hexa4x4x4(xyz_global_i);
  // build another orthonormal basis on it
  basis = Basis(hexa);
  // check orthonormality
  residual = (Integrate([&basis](const Mat3x1& xyz) {
    auto col = basis(xyz);
    A prod = col * col.transpose();
    return prod;
  }, hexa) - A::Identity()).cwiseAbs().maxCoeff();
  EXPECT_NEAR(residual, 0.0, 1e-14);
}
TEST_F(TestHexa4x4x4, Projection) {
  Mat3x8 xyz_global_i;
  xyz_global_i.row(0) << -1, +1, +1, -1, -1, +1, +1, -1;
  xyz_global_i.row(1) << -1, -1, +1, +1, -1, -1, +1, +1;
  xyz_global_i.row(2) << -1, -1, -1, -1, +1, +1, +1, +1;
  auto hexa = Hexa4x4x4(xyz_global_i);
  auto basis = Basis(hexa);
  auto scalar_f = [](Mat3x1 const& xyz){
    return xyz[0] * xyz[1] + xyz[2];
  };
  auto scalar_pf = ScalarPF(scalar_f, basis);
  double residual = (scalar_pf.GetCoef()
      - Mat1x10(0, 0, 0, 1, 0, 1, 0, 0, 0, 0)).cwiseAbs().maxCoeff();
  EXPECT_NEAR(residual, 0.0, 1e-15);
  auto vector_f = [](Mat3x1 const& xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    Mat11x1 func(0, 1,
                x, y, z,
                x * x, x * y, x * z, y * y, y * z, z * z);
    return func;
  };
  auto vector_pf = VectorPF(vector_f, basis);
  Mat11x10 exact_vector;
  exact_vector.row(0).setZero();
  exact_vector.bottomRows(10).setIdentity();
  Mat11x10 abs_diff = vector_pf.GetCoef() - exact_vector;
  EXPECT_NEAR(abs_diff.cwiseAbs().maxCoeff(), 0.0, 1e-14);
}

}  // namespace integrator
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
