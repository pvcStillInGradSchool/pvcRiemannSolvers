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
  auto jump_on_cells = RiemannWithViscosity::IntegrateJumpOnCells();
  std::cout << "[Done] IntegrateJumpOnCells" << std::endl;
  auto jump_on_faces = RiemannWithViscosity::IntegrateJumpOnFaces();
  std::cout << "[Done] IntegrateJumpOnFaces" << std::endl;
  RiemannWithViscosity::SetTimeScale(1.0);
  auto viscosity_values = RiemannWithViscosity::GetViscosityValues(
        jump_on_faces, damping_matrices);
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
  VtkWriter::AddCellData("JumpOnCell1", [&](Cell const &cell) {
    return jump_on_cells.at(cell.id())[0];
  });
  VtkWriter::AddCellData("JumpOnCell2", [&](Cell const &cell) {
    return jump_on_cells.at(cell.id())[1];
  });
  VtkWriter::AddCellData("JumpOnFace1", [&](Cell const &cell) {
    return jump_on_faces.at(cell.id())[0];
  });
  VtkWriter::AddCellData("JumpOnFace2", [&](Cell const &cell) {
    return jump_on_faces.at(cell.id())[1];
  });
  VtkWriter::AddCellData("CellViscosity1", [&](Cell const &cell) {
    return viscosity_values.at(cell.id())[0];
  });
  VtkWriter::AddCellData("CellViscosity2", [&](Cell const &cell) {
    return viscosity_values.at(cell.id())[1];
  });
  VtkWriter::WriteSolutions(part, "Viscosity");
  // Check values by GoogleTest:
  for (auto &curr_cell : part.GetLocalCells()) {
    auto &viscosity_value_on_curr_cell = viscosity_values.at(curr_cell.id());
    for (int k = 0; k < curr_cell.K; k++) {
      EXPECT_LE(0.0, viscosity_value_on_curr_cell[k]);
    }
  }
}

// mpirun -n 4 ./part must be run in ../mesh
// mpirun -n 4 ./viscosity
int main(int argc, char* argv[]) {
  return Main(argc, argv);
}
