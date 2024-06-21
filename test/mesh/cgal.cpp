// Copyright 2023 PEI Weicheng

#include <array>
#include <algorithm>
#include <cstdlib>
#include <vector>

#include "cgnslib.h"
#include "gtest/gtest.h"

#include "mini/mesh/cgal.hpp"


class TestMeshCgal : public ::testing::Test {
 protected:
  double rand() { return -1 + 2.0 * std::rand() / (1.0 + RAND_MAX); }
};
TEST_F(TestMeshCgal, NeighborSearchOnRandomCoordinates) {
  std::srand(31415926);
  int n_source = 1<<10;
  std::vector<double> x(n_source), y(n_source), z(n_source);
  for (int i = 0; i < n_source; ++i) {
    x[i] = rand(); 
    y[i] = rand();
    z[i] = rand();
  }
  auto searching = mini::mesh::cgal::NeighborSearching<double>(x, y, z);
  int n_query = 1<<4;
  int n_neighbor = 8;
  for (int i = 0; i < n_query; ++i) {
    auto a = rand(), b = rand(), c = rand();
    auto cmp = [&x, &y, &z, a, b, c](int i, int j) {
      return std::hypot(x[i] - a, y[i] - b, z[i] - c)
           < std::hypot(x[j] - a, y[j] - b, z[j] - c);
    };
    auto all_indices = std::vector<int>(n_source);
    // TODO(PVC): use std::ranges::view in C++23
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::sort(all_indices.begin(), all_indices.end(), cmp);
    auto neighbor_indices = searching.Search(a, b, c, n_neighbor);
    EXPECT_TRUE(std::equal(neighbor_indices.begin(), neighbor_indices.end(),
        all_indices.begin()));
  }
}
