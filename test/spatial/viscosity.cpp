//  Copyright 2024 PEI Weicheng
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "mini/mesh/cgns.hpp"
#include "mini/mesh/part.hpp"
#include "mini/polynomial/hexahedron.hpp"
#include "mini/polynomial/extrapolation.hpp"

#include "mini/spatial/viscosity.hpp"
#include "mini/spatial/fem.hpp"
#include "mini/spatial/dg/lobatto.hpp"
#include "mini/spatial/fr/general.hpp"
#include "mini/spatial/fr/lobatto.hpp"
#include "mini/basis/vincent.hpp"
#include "mini/input/path.hpp"  // defines PROJECT_BINARY_DIR

#include "test/mesh/part.hpp"

using Polynomial = mini::polynomial::Extrapolation<
    mini::polynomial::Hexahedron<Gx, Gx, Gx, kComponents, true> >;
using Part = mini::mesh::part::Part<cgsize_t, Riemann, Polynomial>;
using Global = typename Part::Global;

auto case_name = PROJECT_BINARY_DIR + std::string("/test/mesh/double_mach");

class TestSpatialViscosity : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetRiemann();
  }
};
TEST_F(TestSpatialViscosity, LobattoFR) {
  auto part = Part(case_name, i_core, n_core);
  InstallPrototype(&part);
  using Spatial = mini::spatial::fr::Lobatto<Part>;
  auto spatial = Spatial(&part);
  using Viscosity = mini::spatial::EnergyBasedViscosity<Part>;
  auto viscosity = Viscosity(&spatial);
  auto damping_matrices = viscosity.BuildDampingMatrices();
  std::cout << "[Done] BuildDampingMatrices" << std::endl;
  auto value_jumps = viscosity.BuildValueJumps();
  std::cout << "[Done] BuildValueJumps" << std::endl;
  auto jump_integrals = viscosity.IntegrateJumps(value_jumps);
  std::cout << "[Done] IntegrateJumps" << std::endl;
  viscosity.TimeScale() = 1e-3;
  auto viscosity_values = viscosity.GetViscosityValues(
        jump_integrals, damping_matrices);
  std::cout << "[Done] GetViscosityValues" << std::endl;
  for (auto &curr_cell : part.GetLocalCells()) {
    auto &value_jumps_on_curr_cell = value_jumps.at(curr_cell.id());
    for (int i_node = 0; i_node < curr_cell.N; ++i_node) {
      auto &value_jumps_on_curr_node = value_jumps_on_curr_cell.at(i_node);
      Global const &global_i = curr_cell.integrator().GetGlobal(i_node);
      Value value_i = curr_cell.polynomial().GetValue(i_node);
      for (auto *neighbor_i : curr_cell.adj_cells_) {
        Value jump_i = value_i - neighbor_i->polynomial().Extrapolate(global_i);
        jump_i = std::abs(jump_i);
        for (int k = 0; k < curr_cell.K; k++) {
          EXPECT_LE(jump_i[k], value_jumps_on_curr_node[k]);
        }
      }
    }
    auto &jump_integral_on_curr_cell = jump_integrals.at(curr_cell.id());
    for (int k = 0; k < curr_cell.K; k++) {
      EXPECT_LE(0.0, jump_integral_on_curr_cell[k]);
    }
    auto &viscosity_on_curr_cell = viscosity_values.at(curr_cell.id());
    std::cout << curr_cell.id() << " " << viscosity_on_curr_cell.transpose() << "\n";
  }
}

// mpirun -n 4 ./part must be run in ../mesh
// mpirun -n 4 ./viscosity
int main(int argc, char* argv[]) {
  return Main(argc, argv);
}
