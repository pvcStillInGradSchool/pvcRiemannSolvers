//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/integrator/function.hpp"
#include "mini/integrator/quad.hpp"

#include "gtest/gtest.h"

using std::sqrt;

namespace mini {
namespace integrator {

class TestQuad4x4 : public ::testing::Test {
 protected:
  using Quad2D4x4 = Quad<double, 2, 4, 4>;
  using Quad3D4x4 = Quad<double, 3, 4, 4>;
  using Mat2x4 = algebra::Matrix<double, 2, 4>;
  using Mat2x1 = algebra::Matrix<double, 2, 1>;
  using Mat3x4 = algebra::Matrix<double, 3, 4>;
  using Mat3x1 = algebra::Matrix<double, 3, 1>;
  using Basis = OrthoNormalBasis<double, 2, 2>;
  using Y = typename Basis::MatNx1;
  using A = typename Basis::MatNxN;
  using Mat1x6 = algebra::Matrix<double, 1, 6>;
  using Mat7x1 = algebra::Matrix<double, 7, 1>;
  using Mat7x6 = algebra::Matrix<double, 7, 6>;
};
TEST_F(TestQuad4x4, VirtualMethods) {
  Mat2x4 xyz_global_i;
  xyz_global_i.row(0) << -1, 1, 1, -1;
  xyz_global_i.row(1) << -1, -1, 1, 1;
  auto quad = Quad2D4x4(xyz_global_i);
  EXPECT_EQ(quad.CountQuadPoints(), 16);
  auto p0 = quad.GetLocalCoord(0);
  EXPECT_EQ(p0[0], -std::sqrt((3 - 2 * std::sqrt(1.2)) / 7));
  EXPECT_EQ(p0[1], -std::sqrt((3 - 2 * std::sqrt(1.2)) / 7));
  auto w1d = (18 + std::sqrt(30)) / 36.0;
  EXPECT_EQ(quad.GetLocalWeight(0), w1d * w1d);
}
TEST_F(TestQuad4x4, In2dSpace) {
  Mat2x4 xyz_global_i;
  xyz_global_i.row(0) << -1, 1, 1, -1;
  xyz_global_i.row(1) << -1, -1, 1, 1;
  auto quad = Quad2D4x4(xyz_global_i);
  static_assert(quad.CellDim() == 2);
  static_assert(quad.PhysDim() == 2);
  EXPECT_EQ(quad.LocalToGlobal(0, 0), Mat2x1(0, 0));
  EXPECT_EQ(quad.LocalToGlobal(1, 1), Mat2x1(1, 1));
  EXPECT_EQ(quad.LocalToGlobal(-1, -1), Mat2x1(-1, -1));
  EXPECT_DOUBLE_EQ(Quadrature([](Mat2x1 const&){ return 2.0; }, quad), 8.0);
  EXPECT_DOUBLE_EQ(Integrate([](Mat2x1 const&){ return 2.0; }, quad), 8.0);
  auto f = [](Mat2x1 const& xy){ return xy[0]; };
  auto g = [](Mat2x1 const& xy){ return xy[1]; };
  auto h = [](Mat2x1 const& xy){ return xy[0] * xy[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, quad), Integrate(h, quad));
  EXPECT_DOUBLE_EQ(Norm(f, quad), sqrt(Innerprod(f, f, quad)));
  EXPECT_DOUBLE_EQ(Norm(g, quad), sqrt(Innerprod(g, g, quad)));
}
TEST_F(TestQuad4x4, In3dSpace) {
  Mat3x4 xyz_global_i;
  xyz_global_i.row(0) << -1, 1, 1, -1;
  xyz_global_i.row(1) << -1, -1, 1, 1;
  xyz_global_i.row(2) << -1, -1, 1, 1;
  auto quad = Quad3D4x4(xyz_global_i);
  static_assert(quad.CellDim() == 2);
  static_assert(quad.PhysDim() == 3);
  EXPECT_EQ(quad.LocalToGlobal(0, 0), Mat3x1(0, 0, 0));
  EXPECT_EQ(quad.LocalToGlobal(1, 1), Mat3x1(1, 1, 1));
  EXPECT_EQ(quad.LocalToGlobal(-1, -1), Mat3x1(-1, -1, -1));
  EXPECT_DOUBLE_EQ(Quadrature([](Mat2x1){ return 2.0; }, quad), 8.0);
  EXPECT_DOUBLE_EQ(Integrate([](Mat3x1){ return 2.0; }, quad), sqrt(2) * 8.0);
  auto f = [](Mat3x1 xyz){ return xyz[0]; };
  auto g = [](Mat3x1 xyz){ return xyz[1]; };
  auto h = [](Mat3x1 xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, quad), Integrate(h, quad));
  EXPECT_DOUBLE_EQ(Norm(f, quad), sqrt(Innerprod(f, f, quad)));
  EXPECT_DOUBLE_EQ(Norm(g, quad), sqrt(Innerprod(g, g, quad)));
}
TEST_F(TestQuad4x4, Basis) {
  Mat2x1 origin = {0, 0}, left = {-1, 2}, right = {1, 3};
  Mat2x4 xyz_global_i;
  xyz_global_i.row(0) << -1, 1, 1, -1;
  xyz_global_i.row(1) << -1, -1, 1, 1;
  auto quad = Quad2D4x4(xyz_global_i);
  auto basis = Basis(quad);
  double residual = (Integrate([&basis](const Mat2x1& xy) {
    auto col = basis(xy);
    A prod = col * col.transpose();
    return prod;
  }, quad) - A::Identity()).cwiseAbs().maxCoeff();
  EXPECT_NEAR(residual, 0.0, 1e-15);
  auto x = left[0], y = left[1];
  xyz_global_i.row(0) << x-1, x+1, x+1, x-1;
  xyz_global_i.row(1) << y-1, y-1, y+1, y+1;
  quad = Quad2D4x4(xyz_global_i);
  basis = Basis(quad);
  residual = (Integrate([&basis](Mat2x1 const& xy) {
    auto col = basis(xy);
    A prod = col * col.transpose();
    return prod;
  }, quad) - A::Identity()).cwiseAbs().maxCoeff();
  EXPECT_NEAR(residual, 0.0, 1e-15);
}
TEST_F(TestQuad4x4, Projection) {
  Mat2x4 xyz_global_i;
  xyz_global_i.row(0) << -1, 1, 1, -1;
  xyz_global_i.row(1) << -1, -1, 1, 1;
  auto quad = Quad2D4x4(xyz_global_i);
  auto basis = Basis(quad);
  auto scalar_f = [](Mat2x1 const& xy){
    return xy[0] * xy[1];
  };
  using ScalarPF = Projection<double, 2, 2, 1>;
  auto scalar_pf = ScalarPF(scalar_f, basis);
  double residual = (scalar_pf.GetCoef()
      - Mat1x6(0, 0, 0, 0, 1, 0)).cwiseAbs().maxCoeff();
  EXPECT_NEAR(residual, 0.0, 1e-15);
  auto vector_f = [](Mat2x1 const& xy) {
    auto x = xy[0], y = xy[1];
    Mat7x1 func(0, 1, x, y, x * x, x * y, y * y);
    return func;
  };
  using VectorPF = Projection<double, 2, 2, 7>;
  auto vector_pf = VectorPF(vector_f, basis);
  Mat7x6 exact_vector{
      {0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0},
      {0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0},
      {0, 0, 0, 0, 0, 1}
  };
  Mat7x6 abs_diff = vector_pf.GetCoef() - exact_vector;
  EXPECT_NEAR(abs_diff.cwiseAbs().maxCoeff(), 0.0, 1e-15);
}

}  // namespace integrator
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
