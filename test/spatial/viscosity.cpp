//  Copyright 2024 PEI Weicheng
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "mini/mesh/cgns.hpp"
#include "mini/mesh/part.hpp"
#include "mini/mesh/vtk.hpp"
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
#include "test/spatial/riemann.hpp"

using Polynomial = mini::polynomial::Extrapolation<
    mini::polynomial::Hexahedron<Gx, Gx, Gx, kComponents, true> >;
using Part = mini::mesh::part::Part<cgsize_t, Polynomial>;
using Global = typename Part::Global;

auto case_name = PROJECT_BINARY_DIR + std::string("/test/mesh/double_mach");

class TestSpatialViscosity : public ::testing::Test {
 protected:
  void SetUp() override {
    test::spatial::ResetRiemann();
  }
};
TEST_F(TestSpatialViscosity, LobattoFR) {
  auto part = Part(case_name, i_core, n_core);
  InstallIntegratorPrototypes(&part);
  part.SetFieldNames({"U1", "U2"});
  using RiemannWithViscosity = mini::spatial::EnergyBasedViscosity<
      Part, test::spatial::Riemann>;
  static_assert(mini::riemann::ConvectiveDiffusive<RiemannWithViscosity>);
  using Spatial = mini::spatial::fr::Lobatto<Part, RiemannWithViscosity>;
  auto spatial = Spatial(&part);
  RiemannWithViscosity::InstallSpatial(&spatial);
  auto damping_matrices = RiemannWithViscosity::BuildDampingMatrices();
  std::cout << "[Done] BuildDampingMatrices" << std::endl;
  // BuildDampingMatrices() modifies Part, so Approximate() is called after it.
  for (auto *cell_ptr : part.GetLocalCellPointers()) {
    cell_ptr->Approximate(func);
  }
  part.ShareGhostCellCoeffs();
  part.UpdateGhostCellCoeffs();
  auto value_jumps = RiemannWithViscosity::BuildValueJumps();
  std::cout << "[Done] BuildValueJumps" << std::endl;
  auto jump_integrals = RiemannWithViscosity::IntegrateJumps(value_jumps);
  std::cout << "[Done] IntegrateJumps" << std::endl;
  RiemannWithViscosity::SetTimeScale(1.0);
  auto viscosity_values = RiemannWithViscosity::GetViscosityValues(
        jump_integrals, damping_matrices);
  std::cout << "[Done] GetViscosityValues" << std::endl;
  RiemannWithViscosity::InitializeRequestsAndBuffers();
  std::cout << "[Done] InitializeRequestsAndBuffers" << std::endl;
  RiemannWithViscosity::ShareGhostCellProperties();
  std::cout << "[Done] ShareGhostCellProperties" << std::endl;
  RiemannWithViscosity::UpdateGhostCellProperties();
  std::cout << "[Done] UpdateGhostCellProperties" << std::endl;
  // Check values by VTK plotting:
  using VtkWriter = mini::mesh::vtk::Writer<Part>;
  using Cell = typename Part::Cell;
  VtkWriter::AddExtraField("CellEnergy1", [&](Cell const &cell, Global const &global, Value const &value){
    return jump_integrals.at(cell.id())[0];
  });
  VtkWriter::AddExtraField("CellEnergy2", [&](Cell const &cell, Global const &global, Value const &value){
    return jump_integrals.at(cell.id())[1];
  });
  VtkWriter::AddExtraField("CellViscosity1", [&](Cell const &cell, Global const &global, Value const &value){
    return viscosity_values.at(cell.id())[0];
  });
  VtkWriter::AddExtraField("CellViscosity2", [&](Cell const &cell, Global const &global, Value const &value){
    return viscosity_values.at(cell.id())[1];
  });
  VtkWriter::WriteSolutions(part, "Viscosity");
  // Check values by GoogleTest:
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
  }
}

// mpirun -n 4 ./part must be run in ../mesh
// mpirun -n 4 ./viscosity
int main(int argc, char* argv[]) {
  return Main(argc, argv);
}
