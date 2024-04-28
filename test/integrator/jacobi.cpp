// Copyright 2023 PEI Weicheng

#include <cmath>
#include <cstdlib>

#include "mini/integrator/jacobi.hpp"
#include "mini/integrator/function.hpp"

#include "gtest/gtest.h"

class TestIntegratorJacobi : public ::testing::Test {
 protected:
  static constexpr int kAlpha{2}, kBeta{0};
  static double rand() {
    return std::rand() / (1.0 + RAND_MAX);
  }
  static double f(double *a, int n, double x) {
    double val = 0.0;
    for (int i = 0; i < n; ++i) {
      val += a[i] * std::pow(1 - x, i);
    }
    return val;
  }
  static double exact(double *a, int n) {
    double val = 0.0;
    for (int i = 0; i < n; ++i) {
      val += a[i] * std::pow(2, i + 3) / (i + 3);
    }
    return val;
  }
};
TEST_F(TestIntegratorJacobi, OnePoint) {
  constexpr int kQuad = 1;
  constexpr int kTerm = 2 * kQuad - 1;
  using Integrator = mini::integrator::Jacobi<double, kQuad, kAlpha, kBeta>;
  std::srand(31415926);
  for (int i = 0; i < 1000; ++i) {
    auto v = std::vector<double>(kTerm);
    for (int j = 0; j < kTerm; ++j) { v[j] = rand(); }
    auto *a = v.data();
    double sum = 0.0;
    for (int i = 0; i < kQuad; ++i) {
      sum += f(a, kTerm, Integrator::points[i]) * Integrator::weights[i];
    }
    EXPECT_NEAR(sum, exact(a, kTerm), 1e-15);
  }
}
TEST_F(TestIntegratorJacobi, TwoPoint) {
  constexpr int kQuad = 2;
  constexpr int kTerm = 2 * kQuad - 1;
  using Integrator = mini::integrator::Jacobi<double, kQuad, kAlpha, kBeta>;
  std::srand(31415926);
  for (int i = 0; i < 1000; ++i) {
    auto v = std::vector<double>(kTerm);
    for (int j = 0; j < kTerm; ++j) { v[j] = rand(); }
    auto *a = v.data();
    double sum = 0.0;
    for (int i = 0; i < kQuad; ++i) {
      sum += f(a, kTerm, Integrator::points[i]) * Integrator::weights[i];
    }
    EXPECT_NEAR(sum, exact(a, kTerm), 1e-14);
  }
}
TEST_F(TestIntegratorJacobi, ThreePoint) {
  constexpr int kQuad = 3;
  constexpr int kTerm = 2 * kQuad - 1;
  using Integrator = mini::integrator::Jacobi<double, kQuad, kAlpha, kBeta>;
  std::srand(31415926);
  for (int i = 0; i < 1000; ++i) {
    auto v = std::vector<double>(kTerm);
    for (int j = 0; j < kTerm; ++j) { v[j] = rand(); }
    auto *a = v.data();
    double sum = 0.0;
    for (int i = 0; i < kQuad; ++i) {
      sum += f(a, kTerm, Integrator::points[i]) * Integrator::weights[i];
    }
    EXPECT_NEAR(sum, exact(a, kTerm), 1e-13);
  }
}
TEST_F(TestIntegratorJacobi, FourPoint) {
  constexpr int kQuad = 4;
  constexpr int kTerm = 2 * kQuad - 1;
  using Integrator = mini::integrator::Jacobi<double, kQuad, kAlpha, kBeta>;
  std::srand(31415926);
  for (int i = 0; i < 1000; ++i) {
    auto v = std::vector<double>(kTerm);
    for (int j = 0; j < kTerm; ++j) { v[j] = rand(); }
    auto *a = v.data();
    double sum = 0.0;
    for (int i = 0; i < kQuad; ++i) {
      sum += f(a, kTerm, Integrator::points[i]) * Integrator::weights[i];
    }
    EXPECT_NEAR(sum, exact(a, kTerm), 1e-12);
  }
}
TEST_F(TestIntegratorJacobi, FivePoint) {
  constexpr int kQuad = 5;
  constexpr int kTerm = 2 * kQuad - 1;
  using Integrator = mini::integrator::Jacobi<double, kQuad, kAlpha, kBeta>;
  std::srand(31415926);
  for (int i = 0; i < 1000; ++i) {
    auto v = std::vector<double>(kTerm);
    for (int j = 0; j < kTerm; ++j) { v[j] = rand(); }
    auto *a = v.data();
    double sum = 0.0;
    for (int i = 0; i < kQuad; ++i) {
      sum += f(a, kTerm, Integrator::points[i]) * Integrator::weights[i];
    }
    EXPECT_NEAR(sum, exact(a, kTerm), 1e-12);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
