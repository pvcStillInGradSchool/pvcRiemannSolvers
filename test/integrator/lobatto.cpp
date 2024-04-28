// Copyright 2019 PEI Weicheng and YANG Minghao

#include <cmath>
#include <cstdlib>

#include "mini/gauss/lobatto.hpp"
#include "mini/gauss/function.hpp"

#include "gtest/gtest.h"

class TestGaussLobatto : public ::testing::Test {
 protected:
  static double rand() {
    return std::rand() / (1.0 + RAND_MAX);
  }
  static double f(double *a, int n, double x) {
    double val = 0.0;
    for (int i = 0; i < n; ++i) {
      val += a[i] * std::pow(x, i);
    }
    return val;
  }
  static double exact(double *a, int n) {
    double val = 0.0;
    for (int i = 0; i < n; ++i) {
      val += a[i] * (i % 2 ? 0.0 : 2.0) / (i + 1.0);
    }
    return val;
  }
};
TEST_F(TestGaussLobatto, TwoPoint) {
  constexpr int kQuad = 2;
  constexpr int kTerm = 2 * kQuad - 3;
  using Gauss = mini::gauss::Lobatto<double, kQuad>;
  static_assert(Gauss::Q == kQuad);
  std::srand(31415926);
  for (int i = 0; i < 1000; ++i) {
    auto v = std::vector<double>(kTerm);
    for (int j = 0; j < kTerm; ++j) { v[j] = rand(); }
    auto *a = v.data();
    double sum = 0.0;
    for (int i = 0; i < kQuad; ++i) {
      sum += f(a, kTerm, Gauss::points[i]) * Gauss::weights[i];
    }
    EXPECT_NEAR(sum, exact(a, kTerm), 1e-15);
  }
}
TEST_F(TestGaussLobatto, ThreePoint) {
  constexpr int kQuad = 3;
  constexpr int kTerm = 2 * kQuad - 3;
  using Gauss = mini::gauss::Lobatto<double, kQuad>;
  static_assert(Gauss::Q == kQuad);
  std::srand(31415926);
  for (int i = 0; i < 1000; ++i) {
    auto v = std::vector<double>(kTerm);
    for (int j = 0; j < kTerm; ++j) { v[j] = rand(); }
    auto *a = v.data();
    double sum = 0.0;
    for (int i = 0; i < kQuad; ++i) {
      sum += f(a, kTerm, Gauss::points[i]) * Gauss::weights[i];
    }
    EXPECT_NEAR(sum, exact(a, kTerm), 1e-15);
  }
}
TEST_F(TestGaussLobatto, FourPoint) {
  constexpr int kQuad = 4;
  constexpr int kTerm = 2 * kQuad - 3;
  using Gauss = mini::gauss::Lobatto<double, kQuad>;
  static_assert(Gauss::Q == kQuad);
  std::srand(31415926);
  for (int i = 0; i < 1000; ++i) {
    auto v = std::vector<double>(kTerm);
    for (int j = 0; j < kTerm; ++j) { v[j] = rand(); }
    auto *a = v.data();
    double sum = 0.0;
    for (int i = 0; i < kQuad; ++i) {
      sum += f(a, kTerm, Gauss::points[i]) * Gauss::weights[i];
    }
    EXPECT_NEAR(sum, exact(a, kTerm), 1e-15);
  }
}
TEST_F(TestGaussLobatto, FivePoint) {
  constexpr int kQuad = 5;
  constexpr int kTerm = 2 * kQuad - 3;
  using Gauss = mini::gauss::Lobatto<double, kQuad>;
  static_assert(Gauss::Q == kQuad);
  std::srand(31415926);
  for (int i = 0; i < 1000; ++i) {
    auto v = std::vector<double>(kTerm);
    for (int j = 0; j < kTerm; ++j) { v[j] = rand(); }
    auto *a = v.data();
    double sum = 0.0;
    for (int i = 0; i < kQuad; ++i) {
      sum += f(a, kTerm, Gauss::points[i]) * Gauss::weights[i];
    }
    EXPECT_NEAR(sum, exact(a, kTerm), 1e-14);
  }
}
TEST_F(TestGaussLobatto, SixPoint) {
  constexpr int kQuad = 6;
  constexpr int kTerm = 2 * kQuad - 3;
  using Gauss = mini::gauss::Lobatto<double, kQuad>;
  static_assert(Gauss::Q == kQuad);
  std::srand(31415926);
  for (int i = 0; i < 1000; ++i) {
    auto v = std::vector<double>(kTerm);
    for (int j = 0; j < kTerm; ++j) { v[j] = rand(); }
    auto *a = v.data();
    double sum = 0.0;
    for (int i = 0; i < kQuad; ++i) {
      sum += f(a, kTerm, Gauss::points[i]) * Gauss::weights[i];
    }
    EXPECT_NEAR(sum, exact(a, kTerm), 1e-14);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
