//  Copyright 2023 PEI Weicheng

#include <cstdlib>

#include "mini/basis/lagrange.hpp"

#include "gtest/gtest.h"

double rand_f() {
  return -1 + 2.0 * std::rand() / (1.0 + RAND_MAX);
}

class TestBasisLagrangeLine : public ::testing::Test {
 protected:
  using Scalar = double;
  using Lagrange = mini::basis::lagrange::Line<Scalar, 4>;
  static_assert(Lagrange::P == 4);
  static_assert(Lagrange::N == 5);

  static Lagrange GetRandomLagrange() {
    std::srand(31415926);
    return Lagrange{ rand_f(), rand_f(), rand_f(), rand_f(), rand_f() };
  }
  static Lagrange GetUniformLagrange() {
    return Lagrange{ -1, -0.5, 0, 0.5 , 1 };
  }
};
TEST_F(TestBasisLagrangeLine, KroneckerDeltaProperty) {
  auto lagrange = GetRandomLagrange();
  for (int i = 0; i < Lagrange::N; ++i) {
    auto x_i = lagrange.GetNode(i);
    auto values = lagrange.GetValues(x_i);
    for (int j = 0; j < Lagrange::N; ++j) {
      EXPECT_EQ(values[j], i == j);
    }
  }
}
TEST_F(TestBasisLagrangeLine, PartitionOfUnityProperty) {
  auto lagrange = GetRandomLagrange();
  for (int i = 1<<10; i >= 0; --i) {
    auto x = rand_f();
    auto values = lagrange.GetValues(x);
    EXPECT_NEAR(values.sum(), 1.0, 1e-14);
  }
}
TEST_F(TestBasisLagrangeLine, GetDerivatives) {
  auto lagrange = GetRandomLagrange();
  // check cached values
  for (int i = 0; i < Lagrange::N; ++i) {
    auto x = lagrange.GetNode(i);
    for (int a = 0; a < Lagrange::N; ++a) {
      EXPECT_EQ(lagrange.GetDerivatives(a, i), lagrange.GetDerivatives(a, x));
    }
  }
  // check with finite differences
  auto delta = 1e-5;
  auto delta2 = delta * delta;
  for (int i = 1<<10; i >= 0; --i) {
    auto x = rand_f();
    auto values_center = lagrange.GetValues(x);
    auto derivatives = lagrange.GetDerivatives(0, x);
    derivatives -= values_center;
    EXPECT_NEAR(derivatives.norm(), 0.0, 1e-11);
    auto values_x_minus = lagrange.GetValues(x - delta);
    auto values_x_plus = lagrange.GetValues(x + delta);
    derivatives = lagrange.GetDerivatives(1, x);
    derivatives -= (values_x_plus - values_x_minus) / (2 * delta);
    EXPECT_NEAR(derivatives.norm(), 0.0, 1e-7);
    derivatives = lagrange.GetDerivatives(2, x);
    derivatives -= ((values_x_minus + values_x_plus) - 2 * values_center) / delta2;
    EXPECT_NEAR(derivatives.norm(), 0.0, 1e-4);
  }
}

class TestBasisLagrangeHexahedron : public ::testing::Test {
 protected:
  using Scalar = double;
  using Lagrange = mini::basis::lagrange::Hexahedron<Scalar, 2, 3, 4>;
  static_assert(Lagrange::N == 3 * 4 * 5);

  static Lagrange GetRandomLagrange() {
    std::srand(31415926);
    return Lagrange {
        Lagrange::LineX{ rand_f(), rand_f(), rand_f() },
        Lagrange::LineY{ rand_f(), rand_f(), rand_f(), rand_f() },
        Lagrange::LineZ{ rand_f(), rand_f(), rand_f(), rand_f(), rand_f() }
    };
  }
};
TEST_F(TestBasisLagrangeHexahedron, KroneckerDeltaProperty) {
  auto lagrange = GetRandomLagrange();
  for (int i = 0; i < Lagrange::N; ++i) {
    auto values = lagrange.GetValues(lagrange.GetNode(i));
    for (int j = 0; j < Lagrange::N; ++j) {
      EXPECT_EQ(values[j], i == j);
    }
  }
}
TEST_F(TestBasisLagrangeHexahedron, PartitionOfUnityProperty) {
  auto lagrange = GetRandomLagrange();
  for (int i = 1<<10; i >= 0; --i) {
    auto x = rand_f(), y = rand_f(), z = rand_f();
    auto values = lagrange.GetValues(x, y, z);
    EXPECT_NEAR(values.sum(), 1.0, 1e-11);
  }
}
TEST_F(TestBasisLagrangeHexahedron, GetDerivatives) {
  auto lagrange = GetRandomLagrange();
  // check cached values
  for (int i = 0; i < Lagrange::I; ++i) {
    for (int j = 0; j < Lagrange::J; ++j) {
      for (int k = 0; k < Lagrange::K; ++k) {
        auto coord = lagrange.GetNode(i, j, k);
        auto x = coord[0], y = coord[1], z = coord[2];
        for (int a = 0; a < Lagrange::I; ++a) {
          for (int b = 0; b < Lagrange::J; ++b) {
            for (int c = 0; c < Lagrange::K; ++c) {
              EXPECT_EQ(lagrange.GetDerivatives(a, b, c, i, j, k),
                        lagrange.GetDerivatives(a, b, c, x, y, z));
            }
          }
        }
      }
    }
  }
  // check with finite differences
  auto delta = 1e-5;
  auto delta2 = delta * delta;
  for (int i = 1<<10; i >= 0; --i) {
    auto x = rand_f(), y = rand_f(), z = rand_f();
    auto values_center = lagrange.GetValues(x, y, z);
    auto derivatives = lagrange.GetDerivatives(0, 0, 0, x, y, z);
    derivatives -= values_center;
    EXPECT_NEAR(derivatives.norm(), 0.0, 1e-9);
    auto values_x_minus = lagrange.GetValues(x - delta, y, z);
    auto values_x_plus = lagrange.GetValues(x + delta, y, z);
    derivatives = lagrange.GetDerivatives(1, 0, 0, x, y, z);
    derivatives -= (values_x_plus - values_x_minus) / (2 * delta);
    EXPECT_NEAR(derivatives.norm(), 0.0, 1e-7);
    auto values_y_minus = lagrange.GetValues(x, y - delta, z);
    auto values_y_plus = lagrange.GetValues(x, y + delta, z);
    derivatives = lagrange.GetDerivatives(0, 1, 0, x, y, z);
    derivatives -= (values_y_plus - values_y_minus) / (2 * delta);
    EXPECT_NEAR(derivatives.norm(), 0.0, 1e-5);
    auto values_z_minus = lagrange.GetValues(x, y, z - delta);
    auto values_z_plus = lagrange.GetValues(x, y, z + delta);
    derivatives = lagrange.GetDerivatives(0, 0, 1, x, y, z);
    derivatives -= (values_z_plus - values_z_minus) / (2 * delta);
    EXPECT_NEAR(derivatives.norm(), 0.0, 1e-5);
    derivatives = lagrange.GetDerivatives(2, 0, 0, x, y, z);
    derivatives -= ((values_x_minus + values_x_plus) - 2 * values_center) / delta2;
    EXPECT_NEAR(derivatives.norm(), 0.0, 1e-1);
    derivatives = lagrange.GetDerivatives(0, 2, 0, x, y, z);
    derivatives -= ((values_y_plus + values_y_minus) - 2 * values_center) / delta2;
    EXPECT_NEAR(derivatives.norm(), 0.0, 1e-1);
    derivatives = lagrange.GetDerivatives(0, 0, 2, x, y, z);
    derivatives -= ((values_z_plus + values_z_minus) - 2 * values_center) / delta2;
    EXPECT_NEAR(derivatives.norm(), 0.0, 1e-1);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
