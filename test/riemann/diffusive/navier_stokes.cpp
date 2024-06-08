// Copyright 2019 PEI Weicheng and YANG Minghao

#include <cstdlib>
#include <vector>

#include "gtest/gtest.h"

#include "mini/constant/index.hpp"
#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/diffusive/navier_stokes.hpp"
#include "mini/coordinate/hexahedron.hpp"
#include "mini/integrator/hexahedron.hpp"
#include "mini/integrator/lobatto.hpp"
#include "mini/polynomial/hexahedron.hpp"

class TestRiemannDiffusiveNavierStokes : public ::testing::Test {
 public:
  using Scalar = double;
  using Gas = mini::riemann::euler::IdealGas<Scalar, 1.4>;
  using NS = mini::riemann::diffusive::NavierStokes<Gas>;
  using Value = typename NS::Value;
  using Primitive = typename NS::Primitive;
  using Conservative = typename NS::Conservative;
  using Gradient = typename NS::Gradient;
  using Vector = typename NS::Vector;
  using Tensor = typename NS::Tensor;
  using FluxMatrix = typename NS::FluxMatrix;
  using FluxVector = typename NS::FluxVector;
  static constexpr int U = 1, V = 2, W = 3, E = 4;
  Scalar nu = 0.01, prandtl = 0.7;

  static double rand_f() {
    return -1 + 2 * std::rand() / (1.0 + RAND_MAX);
  }
  static double disturb(double x) {
    return x * (1 + rand_f() * 0.05);
  }
};
TEST_F(TestRiemannDiffusiveNavierStokes, TestGradientConversions) {
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
  }
}
TEST_F(TestRiemannDiffusiveNavierStokes, TestFluxMatrixFluxVectorConsistency) {
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
    Vector grad_T = primitive_grad_got.col(E);
    // flux_matrix * vector == flux_vector
    FluxMatrix flux_matrix; flux_matrix.setZero();
    FluxVector flux_vector; flux_vector.setZero();
    auto property = NS::GetProperty();
    NS::MinusViscousFlux(&flux_matrix, property,
        conservative_given, conservative_grad_given);
    Vector normal = Vector::Random().normalized();
    EXPECT_NEAR(normal.norm(), 1.0, 1e-15);
    NS::MinusViscousFlux(&flux_vector, property,
        conservative_given, conservative_grad_given, normal);
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
    Scalar value_penalty = 0.0;
    NS::MinusViscousFluxOnNoSlipWall(&flux_vector, property, wall_value,
        conservative_given, conservative_grad_given, normal,
        value_penalty);
    EXPECT_NEAR(flux_vector.norm(), 0.0, 1e-11);
  }
}
TEST_F(TestRiemannDiffusiveNavierStokes, TestViscousStressTensor) {
  using Coordinate = mini::coordinate::Hexahedron8<Scalar>;
  // To approximate quadratic functions in each dimension exactly, at least 3 nodes are needed.
  using IntegratorX = mini::integrator::Lobatto<Scalar, 3>;
  using Interpolation = mini::polynomial::Hexahedron<IntegratorX, IntegratorX, IntegratorX,
      NS::kComponents, true>;
  using Basis = typename Interpolation::Basis;
  using Integrator = typename Interpolation::Integrator;
  using Coeff = typename Interpolation::Coeff;
  using Value = typename Interpolation::Value;
  using Global = typename Integrator::Global;
  // build a hexa-integrator and a Lagrange basis on it
  auto a = 2.0, b = 3.0, c = 4.0;
  auto coordinate = Coordinate {
    Global(-a, -b, -c), Global(+a, -b, -c),
    Global(+a, +b, -c), Global(-a, +b, -c),
    Global(-a, -b, +c), Global(+a, -b, +c),
    Global(+a, +b, +c), Global(-a, +b, +c),
  };
  auto integrator = Integrator(coordinate);
  auto interp = Interpolation(integrator);
  // build a vector function and its interpolation
  Scalar rho = 1.29, p = 101325;
  NS::SetProperty(nu, prandtl);
  auto [mu, zeta] = NS::GetViscosity(rho, nu);
  std::srand(31415926);
  Scalar u_0 = rand_f(), du_dx = rand_f(), du_dy = rand_f(), du_dz = rand_f();
  Scalar v_0 = rand_f(), dv_dx = rand_f(), dv_dy = rand_f(), dv_dz = rand_f();
  Scalar w_0 = rand_f(), dw_dx = rand_f(), dw_dy = rand_f(), dw_dz = rand_f();
  Scalar div_uvw = du_dx + dv_dy + dw_dz;
  EXPECT_GT(std::abs(du_dx), 1e-2);
  EXPECT_GT(std::abs(du_dy), 1e-2);
  EXPECT_GT(std::abs(du_dz), 1e-2);
  EXPECT_GT(std::abs(dv_dx), 1e-2);
  EXPECT_GT(std::abs(dv_dy), 1e-2);
  EXPECT_GT(std::abs(dv_dz), 1e-2);
  EXPECT_GT(std::abs(dw_dx), 1e-2);
  EXPECT_GT(std::abs(dw_dy), 1e-2);
  EXPECT_GT(std::abs(dw_dz), 1e-2);
  using namespace mini::constant::index;
  auto func = [&](Global const& xyz) {
    Scalar x = xyz[X], y = xyz[Y], z = xyz[Z];
    Scalar u = u_0 + du_dx * x + du_dy * y + du_dz * z;
    Scalar v = v_0 + dv_dx * x + dv_dy * y + dv_dz * z;
    Scalar w = w_0 + dw_dx * x + dw_dy * y + dw_dz * z;
    Primitive primitive{ rho, u, v, w , p };
    return Gas::PrimitiveToConservative(primitive);
  };
  interp.Approximate(func);
  // test values on nodes
  for (int ijk = 0; ijk < Basis::N; ++ijk) {
    auto [conservative_value, conservative_grad] =
        interp.GetGlobalValueGradient(ijk);
    auto [pimitive_value, primitive_grad] = NS::ConservativeToPrimitive(
        conservative_value, conservative_grad);
    Tensor tau = NS::GetViscousStressTensor(primitive_grad, rho, nu);
    EXPECT_NEAR(tau[XY], mu * (du_dy + dv_dx), 1e-16);
    EXPECT_NEAR(tau[YZ], mu * (dv_dz + dw_dy), 1e-16);
    EXPECT_NEAR(tau[ZX], mu * (dw_dx + du_dz), 1e-16);
    EXPECT_NEAR(tau[XX], mu * (du_dx * 4./3 - (dv_dy + dw_dz) * 2./3), 1e-16);
    EXPECT_NEAR(tau[YY], mu * (dv_dy * 4./3 - (du_dx + dw_dz) * 2./3), 1e-16);
    EXPECT_NEAR(tau[ZZ], mu * (dw_dz * 4./3 - (du_dx + dv_dy) * 2./3), 1e-16);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
