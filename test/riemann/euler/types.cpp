// Copyright 2019 PEI Weicheng and YANG Minghao

#include <cstdlib>
#include <vector>

#include "gtest/gtest.h"

#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/diffusive/navier_stokes.hpp"

double rand_f() {
  return -1 + 2 * std::rand() / (1.0 + RAND_MAX);
}
double disturb(double x) {
  return x * (1 + rand_f() * 0.05);
}

namespace mini {
namespace riemann {
namespace euler {

class TestTypes : public ::testing::Test {
};
TEST_F(TestTypes, TestTuples) {
  auto rho{0.1}, u{+0.3}, v{-0.4}, p{0.5};
  auto primitive = Primitives<double, 2>{rho, u, v, p};
  EXPECT_DOUBLE_EQ(primitive[0], primitive.rho());
  EXPECT_DOUBLE_EQ(primitive.rho(), primitive.mass());
  EXPECT_DOUBLE_EQ(primitive[1], primitive.u());
  EXPECT_DOUBLE_EQ(primitive.u(), primitive.momentumX());
  EXPECT_DOUBLE_EQ(primitive[2], primitive.v());
  EXPECT_DOUBLE_EQ(primitive.v(), primitive.momentumY());
  EXPECT_DOUBLE_EQ(primitive[3], primitive.p());
  EXPECT_DOUBLE_EQ(primitive.p(), primitive.energy());
  using Vector = typename Primitives<double, 2>::Vector;
  EXPECT_EQ(primitive.momentum(), Vector(+0.3, -0.4));
  EXPECT_DOUBLE_EQ(primitive.GetDynamicPressure(), rho * (u*u + v*v) / 2);
}
TEST_F(TestTypes, TestIdealGasProperties) {
  using Scalar = double;
  Scalar constexpr kGamma = 1.4;
  using Gas = IdealGas<Scalar, kGamma>;
  Scalar density = 1.293;
  Scalar pressure = 101325;
  Scalar temperature = pressure / density / Gas::R();
  EXPECT_NEAR(temperature, 273.15, 1e-0);
  EXPECT_EQ(Gas::GetSpeedOfSound(temperature),
      Gas::GetSpeedOfSound(density, pressure));
  Scalar mach = 0.2;
  Scalar factor = 1 + Gas::GammaMinusOneOverTwo() * mach * mach;
  Scalar total_temperature = temperature * factor;
  EXPECT_EQ(Gas::TotalTemperatureToTemperature(mach, total_temperature),
      temperature);
  Scalar total_pressure = pressure *
      std::pow(factor, Gas::GammaOverGammaMinusOne());
  EXPECT_EQ(Gas::TotalPressureToPressure(mach, total_pressure),
      pressure);
  EXPECT_NEAR(Gas::Cp() / Gas::Cv(), kGamma, 1e-15);
  EXPECT_NEAR(Gas::Cp(), 1005, 1e0);
  EXPECT_NEAR(Gas::GetMachFromPressure(pressure, total_pressure),
      mach, 1e-16);
  EXPECT_NEAR(Gas::GetMachFromTemperature(temperature, total_temperature),
      mach, 1e-16);
}
TEST_F(TestTypes, TestIdealGasViscosity) {
  using Scalar = double;
  Scalar constexpr kGamma = 1.4;
  using Gas = IdealGas<Scalar, kGamma>;
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
    EXPECT_NEAR(normal.dot(flux_matrix.row(V)), flux_vector[V], 1e-16);
    EXPECT_NEAR(normal.dot(flux_matrix.row(W)), flux_vector[W], 1e-15);
    EXPECT_NEAR(normal.dot(flux_matrix.row(E)), flux_vector[E], 1e-12);
    EXPECT_NEAR(normal.dot(flux_matrix.row(U)), flux_vector[U], 1e-16);
    EXPECT_NEAR(normal.dot(flux_matrix.row(V)), flux_vector[V], 1e-16);
    EXPECT_NEAR(normal.dot(flux_matrix.row(W)), flux_vector[W], 1e-15);
    EXPECT_NEAR(normal.dot(flux_matrix.row(E)), flux_vector[E], 1e-12);
  }
}
TEST_F(TestTypes, TestConverters) {
  auto rho{0.1}, u{+0.2}, v{-0.2}, p{0.3};
  auto primitive = Primitives<double, 2>{rho, u, v, p};
  using Gas = IdealGas<double, 1.4>;
  constexpr auto gamma = Gas::Gamma();
  auto conservative = Conservatives<double, 2>{
    rho, rho*u, rho*v, p/(gamma-1) + 0.5*rho*(u*u + v*v)
  };
  EXPECT_EQ(Gas::PrimitiveToConservative(primitive), conservative);
  auto primitive_copy = Gas::ConservativeToPrimitive(conservative);
  EXPECT_DOUBLE_EQ(primitive_copy.rho(), rho);
  EXPECT_DOUBLE_EQ(primitive_copy.u(), u);
  EXPECT_DOUBLE_EQ(primitive_copy.v(), v);
  EXPECT_DOUBLE_EQ(primitive_copy.p(), p);
}

}  // namespace euler
}  // namespace riemann
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
