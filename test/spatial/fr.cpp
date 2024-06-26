//  Copyright 2023 PEI Weicheng
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "mini/mesh/cgns.hpp"
#include "mini/mesh/part.hpp"
#include "mini/polynomial/hexahedron.hpp"

#define ENABLE_LOGGING
#include "mini/spatial/fr/general.hpp"
#include "mini/spatial/fr/lobatto.hpp"
#include "mini/basis/vincent.hpp"
#include "mini/input/path.hpp"  // defines PROJECT_BINARY_DIR

#include "test/mesh/part.hpp"
#include "test/spatial/riemann.hpp"

using Polynomial = mini::polynomial::Hexahedron<Gx, Gx, Gx, kComponents, true>;
using Part = mini::mesh::part::Part<cgsize_t, Polynomial>;

auto case_name = PROJECT_BINARY_DIR + std::string("/test/mesh/double_mach");

class TestSpatialFR : public ::testing::Test {
 protected:
  void SetUp() override;
};
void TestSpatialFR::SetUp() {
  test::spatial::ResetRiemann();
}
template <typename Spatial>
auto GetResidualColumn(Spatial *spatial_ptr, Part *part_ptr) {
  spatial_ptr->SetSmartBoundary("4_S_31", moving);  // Left
  spatial_ptr->SetInviscidWall("4_S_1");   // Back
  spatial_ptr->SetSubsonicInlet("4_S_32", moving);  // Front
  spatial_ptr->SetSubsonicOutlet("4_S_23", moving);  // Right
  spatial_ptr->SetSupersonicInlet("4_S_27", moving);  // Top
  spatial_ptr->SetSupersonicOutlet("4_S_15");  // Gap
  spatial_ptr->SetSupersonicOutlet("4_S_19");  // Bottom
  for (auto *cell_ptr : part_ptr->GetLocalCellPointers()) {
    cell_ptr->Approximate(func);
  }
  spatial_ptr->SetTime(1.5);
  std::printf("%s() proc[%d/%d] cost %f sec\n",
      spatial_ptr->name().c_str(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  auto column = spatial_ptr->GetSolutionColumn();
  assert(column.size() == part_ptr->GetCellDataSize());
  double global_norm2, local_norm2 = column.squaredNorm();
  spatial_ptr->SetSolutionColumn(column);
  std::printf("solution.squaredNorm() == %6.2e on proc[%d/%d] cost %f sec\n",
      local_norm2, i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  MPI_Reduce(&local_norm2, &global_norm2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (i_core == 0) {
    std::printf("solution.squaredNorm() == %6.2e on proc[all] cost %f sec\n",
        global_norm2, MPI_Wtime() - time_begin);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  column = spatial_ptr->GetResidualColumn();
  local_norm2 = column.squaredNorm();
  std::printf("residual.squaredNorm() == %6.2e on proc[%d/%d] cost %f sec\n",
      local_norm2, i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  MPI_Reduce(&local_norm2, &global_norm2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (i_core == 0) {
    std::printf("residual.squaredNorm() == %6.2e on proc[all] cost %f sec\n",
        global_norm2, MPI_Wtime() - time_begin);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  return column;
}
TEST_F(TestSpatialFR, CompareResiduals) {
  auto part = Part(case_name, i_core, n_core);
  InstallIntegratorPrototypes(&part);
  /* aproximated by Lagrange basis on Lobatto roots with general correction functions */
  time_begin = MPI_Wtime();
  using General = mini::spatial::fr::General<Part, test::spatial::Riemann>;
  using Vincent = mini::basis::Vincent<Scalar>;
  auto general = General(&part, Vincent::HuynhLumpingLobatto(kDegrees));
  auto general_residual = GetResidualColumn(&general, &part);
  /* aproximated by Lagrange basis on Lobatto roots with Huynh's correction functions */
  time_begin = MPI_Wtime();
  using Lobatto = mini::spatial::fr::Lobatto<Part, test::spatial::Riemann>;
  auto lobatto = Lobatto(&part);
  auto lobatto_residual = GetResidualColumn(&lobatto, &part);
  EXPECT_NEAR(0, (general_residual - lobatto_residual).squaredNorm(), 1e-15);
}

// mpirun -n 4 ./part must be run in ../mesh
// mpirun -n 4 ./fr
int main(int argc, char* argv[]) {
  return Main(argc, argv);
}
