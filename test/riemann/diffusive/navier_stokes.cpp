// Copyright 2019 PEI Weicheng and YANG Minghao

#include <cstdlib>
#include <vector>

#include "gtest/gtest.h"

#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/diffusive/navier_stokes.hpp"

class TestRiemannDiffusiveNavierStokes : public ::testing::Test {
 public:
  static double rand_f() {
    return -1 + 2 * std::rand() / (1.0 + RAND_MAX);
  }
  static double disturb(double x) {
    return x * (1 + rand_f() * 0.05);
  }
};
TEST_F(TestRiemannDiffusiveNavierStokes, TestIdealGasViscosity) {
  using Scalar = double;
  Scalar constexpr kGamma = 1.4;
  using Gas = mini::riemann::euler::IdealGas<Scalar, kGamma>;
  using NS = mini::riemann::diffusive::NavierStokes<Gas>;
  using Primitive = typename NS::Primitive;
  using Conservative = typename NS::Conservative;
  using Gradient = typename NS::Gradient;
  using Vector = typename NS::Vector;
  using Tensor = typename NS::Tensor;
  using FluxMatrix = typename NS::FluxMatrix;
  using FluxVector = typename NS::FluxVector;
  Scalar nu = 0.01, prandtl = 0.7;
  NS::SetProperty(nu, prandtl);
  std::srand(31415926);
  for (int i = 1 << 10; i >= 0; --i) {
    Scalar rho = disturb(1.29);
    Scalar u = disturb(10.0);
    Scalar v = disturb(20.0);
    Scalar w = disturb(30.0);
    Scalar p = disturb(101325);
    auto primitive_given = Primitive(rho, u, v, w, p);
    auto conservative_given = Gas::PrimitiveToConservative(primitive_given);
    Gradient conservative_grad_given = Gradient::Random();
    auto [primitive_got, primitive_grad_got] = NS::ConservativeToPrimitive(
        conservative_given, conservative_grad_given);
    EXPECT_NEAR((primitive_got - primitive_given).norm(), 0.0, 1e-10);
    Vector grad_rho = conservative_grad_given.col(0);
    constexpr int U = 1, V = 2, W = 3, E = 4;
    Vector grad_got;
    // grad(rho * u)
    Vector grad_u = primitive_grad_got.col(U);
    grad_got = u * grad_rho + rho * grad_u;
    EXPECT_NEAR((grad_got - conservative_grad_given.col(U)).norm(), 0.0, 1e-14);
    // grad(rho * v)
    Vector grad_v = primitive_grad_got.col(V);
    grad_got = v * grad_rho + rho * grad_v;
    EXPECT_NEAR((grad_got - conservative_grad_given.col(V)).norm(), 0.0, 1e-14);
    // grad(rho * w)
    Vector grad_w = primitive_grad_got.col(W);
    grad_got = w * grad_rho + rho * grad_w;
    EXPECT_NEAR((grad_got - conservative_grad_given.col(W)).norm(), 0.0, 1e-13);
    // grad(rho * Cv * T + rho * (u * u + v * v + w * w) / 2)
    Vector grad_T = primitive_grad_got.col(E);
    Scalar T = p / Gas::R() / rho;
    grad_got = (grad_rho * T + rho * grad_T) * Gas::Cv();
    grad_got += grad_rho * (u * u + v * v + w * w) / 2;
    grad_got += rho * (grad_u * u + grad_v * v + grad_w * w);
    EXPECT_NEAR((grad_got - conservative_grad_given.col(E)).norm(), 0.0, 1e-9);
    // flux_matrix * vector == flux_vector
    FluxMatrix flux_matrix; flux_matrix.setZero();
    FluxVector flux_vector; flux_vector.setZero();
    NS::MinusViscousFlux(conservative_given, conservative_grad_given,
        &flux_matrix);
    Vector normal = Vector::Random().normalized();
    EXPECT_NEAR(normal.norm(), 1.0, 1e-15);
    NS::MinusViscousFlux(conservative_given, conservative_grad_given, normal,
        &flux_vector);
    EXPECT_EQ(flux_vector[0], 0.0);
    EXPECT_NE(flux_vector[U], 0.0);
    EXPECT_NE(flux_vector[V], 0.0);
    EXPECT_NE(flux_vector[W], 0.0);
    EXPECT_NE(flux_vector[E], 0.0);
    EXPECT_NEAR(normal.dot(flux_matrix.row(U)), flux_vector[U], 1e-16);
    EXPECT_NEAR(normal.dot(flux_matrix.row(V)), flux_vector[V], 1e-16);
    EXPECT_NEAR(normal.dot(flux_matrix.row(W)), flux_vector[W], 1e-15);
    EXPECT_NEAR(normal.dot(flux_matrix.row(E)), flux_vector[E], 1e-12);
    Primitive wall_value = primitive_got;
    wall_value.energy() = normal.dot(grad_T);
    flux_vector = -flux_vector;
    NS::MinusViscousFluxOnSlidingWall(wall_value,
        conservative_given, conservative_grad_given, normal, &flux_vector);
    EXPECT_NEAR(flux_vector.norm(), 0.0, 1e-11);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
