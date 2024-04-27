//  Copyright 2024 PEI Weicheng
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "mini/mesh/cgns.hpp"
#include "mini/mesh/part.hpp"
#include "mini/polynomial/hexahedron.hpp"

#include "mini/spatial/viscosity.hpp"
#include "mini/spatial/fem.hpp"
#include "mini/spatial/dg/lobatto.hpp"
#include "mini/spatial/fr/general.hpp"
#include "mini/spatial/fr/lobatto.hpp"
#include "mini/basis/vincent.hpp"
#include "mini/input/path.hpp"  // defines PROJECT_BINARY_DIR

#include "test/mesh/part.hpp"

using Projection = mini::polynomial::Hexahedron<Gx, Gx, Gx, kComponents, true>;
using Part = mini::mesh::part::Part<cgsize_t, Riemann, Projection>;

auto case_name = PROJECT_BINARY_DIR + std::string("/test/mesh/double_mach");

class TestSpatialViscosity : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetRiemann();
  }
};
TEST_F(TestSpatialViscosity, LobattoFR) {
  auto part = Part(case_name, i_core, n_core);
  using Spatial = mini::spatial::fr::Lobatto<Part>;
  auto spatial = Spatial(&part);
  using Viscosity = mini::spatial::EnergyBasedViscosity<Part>;
  auto viscosity = Viscosity(&spatial);
  // auto matrices = viscosity.BuildDampingMatrices();
  std::cout << "[Done] BuildDampingMatrices" << std::endl;
  auto local_on_neighbors = viscosity.BuildCoordinates();
  std::cout << "[Done] BuildCoordinates" << std::endl;
  for (auto &cell_i : part.GetLocalCells()) {
    int i_cell = cell_i.id();
    auto &local_on_neighbors_of_cell_i = local_on_neighbors.at(i_cell);
    int n_neighbor = cell_i.adj_cells_.size();
    for (int i_neighbor = 0; i_neighbor < n_neighbor; ++i_neighbor) {
      auto &local_on_neighbor_i = local_on_neighbors_of_cell_i.at(i_neighbor);
      auto *neighbor_i = cell_i.adj_cells_.at(i_neighbor);
      for (int i_node = 0; i_node < cell_i.N; ++i_node) {
        auto &global = cell_i.gauss().GetGlobal(i_node);
        auto global_from_neighbor
            = neighbor_i->coordinate().LocalToGlobal(local_on_neighbor_i.at(i_node));
        EXPECT_NEAR((global - global_from_neighbor).norm(), 0, 1e-10);
      }
    }
  }
}

// mpirun -n 4 ./part must be run in ../mesh
// mpirun -n 4 ./viscosity
int main(int argc, char* argv[]) {
  return Main(argc, argv);
}
