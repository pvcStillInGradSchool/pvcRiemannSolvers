//  Copyright 2024 PEI Weicheng

#include <cstdlib>
#include <vector>

#include "mini/rand.hpp"

#include "gtest/gtest.h"

class TestRand : public ::testing::Test {
 protected:
  static constexpr double a = -1.0, b = 1.0;
  static constexpr int kTrails = 1 << 10;
  static constexpr unsigned int seed = 31415926;
};
TEST_F(TestRand, SameSequence) {
  auto BuildSequence = [](unsigned int seed) {
    std::vector<double> vals;
    for (int k = 0; k < kTrails; ++k) {
      vals.push_back(mini::rand::uniform_r(a, b, &seed));
    }
    return vals;
  };
  EXPECT_EQ(BuildSequence(seed), BuildSequence(seed));
}
