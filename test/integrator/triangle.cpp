//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/integrator/function.hpp"
#include "mini/integrator/triangle.hpp"

#include "gtest/gtest.h"

class TestTriangleIntegrator : public ::testing::Test {
 protected:
  using Triangle2D = mini::integrator::Triangle<double, 2, 16>;
  using Triangle3D = mini::integrator::Triangle<double, 3, 16>;
  using Mat2x3 = mini::algebra::Matrix<double, 2, 3>;
  using Mat2x1 = mini::algebra::Matrix<double, 2, 1>;
  using Mat3x3 = mini::algebra::Matrix<double, 3, 3>;
  using Mat3x1 = mini::algebra::Matrix<double, 3, 1>;
};
TEST_F(TestTriangleIntegrator, VirtualMethods) {
  Mat2x3 xy_global_i;
  xy_global_i.row(0) << 0, 2, 2;
  xy_global_i.row(1) << 0, 0, 2;
  auto tri = Triangle2D(xy_global_i);
  EXPECT_EQ(tri.CountQuadraturePoints(), 16);
}
TEST_F(TestTriangleIntegrator, In2dSpace) {
  Mat2x3 xy_global_i;
  xy_global_i.row(0) << 0, 2, 2;
  xy_global_i.row(1) << 0, 0, 2;
  auto tri = Triangle2D(xy_global_i);
  static_assert(tri.CellDim() == 2);
  static_assert(tri.PhysDim() == 2);
  EXPECT_DOUBLE_EQ(tri.area(), 2.0);
  EXPECT_EQ(tri.center(), Mat2x1(4./3, 2./3));
  EXPECT_EQ(tri.LocalToGlobal(1, 0), Mat2x1(0, 0));
  EXPECT_EQ(tri.LocalToGlobal(0, 1), Mat2x1(2, 0));
  EXPECT_EQ(tri.LocalToGlobal(0, 0), Mat2x1(2, 2));
  EXPECT_DOUBLE_EQ(Quadrature([](Mat2x1 const&){ return 2.0; }, tri), 1.0);
  EXPECT_DOUBLE_EQ(Integrate([](Mat2x1 const&){ return 2.0; }, tri), 4.0);
  auto f = [](Mat2x1 const& xy){ return xy[0]; };
  auto g = [](Mat2x1 const& xy){ return xy[1]; };
  auto h = [](Mat2x1 const& xy){ return xy[0] * xy[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, tri), Integrate(h, tri));
  EXPECT_DOUBLE_EQ(Norm(f, tri), std::sqrt(Innerprod(f, f, tri)));
  EXPECT_DOUBLE_EQ(Norm(g, tri), std::sqrt(Innerprod(g, g, tri)));
}
TEST_F(TestTriangleIntegrator, In3dSpace) {
  Mat3x3 xyz_global_i;
  xyz_global_i.row(0) << 0, 2, 2;
  xyz_global_i.row(1) << 0, 0, 2;
  xyz_global_i.row(2) << 2, 2, 2;
  auto tri = Triangle3D(xyz_global_i);
  static_assert(tri.CellDim() == 2);
  static_assert(tri.PhysDim() == 3);
  EXPECT_DOUBLE_EQ(tri.area(), 2.0);
  EXPECT_EQ(tri.center(), Mat3x1(4./3, 2./3, 2.));
  EXPECT_EQ(tri.LocalToGlobal(1, 0), Mat3x1(0, 0, 2));
  EXPECT_EQ(tri.LocalToGlobal(0, 1), Mat3x1(2, 0, 2));
  EXPECT_EQ(tri.LocalToGlobal(0, 0), Mat3x1(2, 2, 2));
  EXPECT_DOUBLE_EQ(
      Quadrature([](Mat2x1 const&){ return 2.0; }, tri), 1.0);
  EXPECT_DOUBLE_EQ(
      Integrate([](Mat3x1 const&){ return 2.0; }, tri), 4.0);
  auto f = [](Mat3x1 const& xyz){ return xyz[0]; };
  auto g = [](Mat3x1 const& xyz){ return xyz[1]; };
  auto h = [](Mat3x1 const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, tri), Integrate(h, tri));
  EXPECT_DOUBLE_EQ(Norm(f, tri), std::sqrt(Innerprod(f, f, tri)));
  EXPECT_DOUBLE_EQ(Norm(g, tri), std::sqrt(Innerprod(g, g, tri)));
  // test normal frames
  tri.BuildNormalFrames();
  for (int q = 0; q < tri.CountQuadraturePoints(); ++q) {
    auto& frame = tri.GetNormalFrame(q);
    auto &nu = frame.col(0), &sigma = frame.col(1), &pi = frame.col(2);
    EXPECT_EQ(nu, sigma.cross(pi));
    EXPECT_EQ(sigma, pi.cross(nu));
    EXPECT_EQ(pi, nu.cross(sigma));
    EXPECT_EQ(nu, Mat3x1(0, 0, 1));
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}