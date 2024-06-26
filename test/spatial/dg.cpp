//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "mpi.h"
#include "pcgnslib.h"

#include "mini/mesh/cgns.hpp"
#include "mini/mesh/part.hpp"
#include "mini/mesh/vtk.hpp"
#include "mini/limiter/weno.hpp"
#include "mini/limiter/reconstruct.hpp"
#include "mini/polynomial/projection.hpp"
#include "mini/polynomial/hexahedron.hpp"
#include "mini/spatial/fem.hpp"
#include "mini/spatial/dg/general.hpp"
#include "mini/spatial/dg/lobatto.hpp"
#include "mini/spatial/with_limiter.hpp"
#include "mini/temporal/ode.hpp"
#include "mini/input/path.hpp"  // defines PROJECT_BINARY_DIR

#include "test/mesh/part.hpp"
#include "test/spatial/riemann.hpp"

mini::temporal::Euler<Scalar> temporal;
double t_curr = 1.5, dt = 1e-3;

template <class Part>
Scalar Norm1(Part const &part) {
  auto norm_1 = 0.0;
  for (auto &cell : part.GetLocalCells()) {
    for (int i = 0, n = cell.integrator().CountPoints(); i < n; ++i) {
      auto v = cell.polynomial().GetValue(i);
      norm_1 += std::abs(v[0]) + std::abs(v[1]);
    }
  }
  return norm_1;
}

auto case_name = PROJECT_BINARY_DIR + std::string("/test/mesh/double_mach");

// mpirun -n 4 ./part must be run in ../mesh
// mpirun -n 4 ./dg
int main(int argc, char* argv[]) {
  int n_core, i_core;
  double time_begin;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &n_core);
  MPI_Comm_rank(MPI_COMM_WORLD, &i_core);
  cgp_mpi_comm(MPI_COMM_WORLD);

  test::spatial::ResetRiemann();

  /* aproximated by Projection on OrthoNormal basis */
{
  time_begin = MPI_Wtime();
  using Polynomial = mini::polynomial::Projection<
      Scalar, kDimensions, kDegrees, kComponents>;
  using Part = mini::mesh::part::Part<cgsize_t, Polynomial>;
  auto part = Part(case_name, i_core, n_core);
  InstallIntegratorPrototypes(&part);
  using Cell = typename Part::Cell;
  using Limiter = mini::limiter::weno::Lazy<Cell>;
  auto limiter = Limiter(/* w0 = */0.001, /* eps = */1e-6);
  using General = mini::spatial::dg::General<Part, test::spatial::Riemann>;
  using Spatial = mini::spatial::WithLimiter<General, Limiter>;
  auto spatial = Spatial(&limiter, &part);
  spatial.SetSmartBoundary("4_S_27", moving);  // Top
  spatial.SetSmartBoundary("4_S_31", moving);  // Left
  spatial.SetInviscidWall("4_S_1");   // Back
  spatial.SetSubsonicInlet("4_S_32", moving);  // Front
  spatial.SetSupersonicInlet("4_S_19", moving);  // Bottom
  spatial.SetSubsonicOutlet("4_S_23", moving);  // Right
  spatial.SetSupersonicOutlet("4_S_15");  // Gap
  for (Cell *cell_ptr : part.GetLocalCellPointers()) {
    cell_ptr->Approximate(func);
  }
  mini::limiter::Reconstruct(&part, &limiter);
  spatial.SetTime(1.5);
  std::printf("Part on basis::OrthoNormal proc[%d/%d] cost %f sec\n",
      i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  auto column = spatial.GetSolutionColumn();
  assert(column.size() == part.GetCellDataSize());
  spatial.SetSolutionColumn(column);
  column -= spatial.GetSolutionColumn();
  std::printf("solution.squaredNorm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.squaredNorm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  column = spatial.GetResidualColumn();
  std::printf("residual.squaredNorm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.squaredNorm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);
}
  /* aproximated by Projection on Lagrange basis on Lobatto roots */
{
  time_begin = MPI_Wtime();
  /* Check equivalence between local and global formulation. */
{
  using Polynomial = mini::polynomial::Hexahedron<Gx, Gx, Gx, kComponents, true>;
  using Part = mini::mesh::part::Part<cgsize_t, Polynomial>;
  using Spatial = mini::spatial::dg::Lobatto<Part, test::spatial::Riemann>;
  auto part = Part(case_name, i_core, n_core);
  InstallIntegratorPrototypes(&part);
  auto spatial = Spatial(&part);
  spatial.SetSmartBoundary("4_S_27", moving);  // Top
  spatial.SetSmartBoundary("4_S_31", moving);  // Left
  spatial.SetInviscidWall("4_S_1");   // Back
  spatial.SetSubsonicInlet("4_S_32", moving);  // Front
  spatial.SetSupersonicInlet("4_S_19", moving);  // Bottom
  spatial.SetSubsonicOutlet("4_S_23", moving);  // Right
  spatial.SetSupersonicOutlet("4_S_15");  // Gap
  for (auto *cell_ptr : part.GetLocalCellPointers()) {
    cell_ptr->Approximate(func);
  }
  spatial.SetTime(1.5);
  std::printf("Part on basis::lagrange::Hexahedron<Local> proc[%d/%d] cost %f sec\n",
      i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  auto column = spatial.GetSolutionColumn();
  assert(column.size() == part.GetCellDataSize());
  spatial.SetSolutionColumn(column);
  column -= spatial.GetSolutionColumn();
  std::printf("solution.squaredNorm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.squaredNorm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  column = spatial.GetResidualColumn();
  std::printf("residual.squaredNorm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.squaredNorm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  std::printf("val_curr.squaredNorm() == %6.2e on proc[%d/%d] cost %f sec\n",
      Norm1(part), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  temporal.Update(&spatial, t_curr, dt);
  std::printf("val_next.squaredNorm() == %6.2e on proc[%d/%d] cost %f sec\n",
      Norm1(part), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);
}
  using Polynomial = mini::polynomial::Hexahedron<Gx, Gx, Gx, kComponents>;
  using Part = mini::mesh::part::Part<cgsize_t, Polynomial>;
  using Spatial = mini::spatial::dg::Lobatto<Part, test::spatial::Riemann>;
  auto part = Part(case_name, i_core, n_core);
  InstallIntegratorPrototypes(&part);
  auto spatial = Spatial(&part);
  spatial.SetSmartBoundary("4_S_27", moving);  // Top
  spatial.SetSmartBoundary("4_S_31", moving);  // Left
  spatial.SetInviscidWall("4_S_1");   // Back
  spatial.SetSubsonicInlet("4_S_32", moving);  // Front
  spatial.SetSupersonicInlet("4_S_19", moving);  // Bottom
  spatial.SetSubsonicOutlet("4_S_23", moving);  // Right
  spatial.SetSupersonicOutlet("4_S_15");  // Gap
  for (auto *cell_ptr : part.GetLocalCellPointers()) {
    cell_ptr->Approximate(func);
  }
  spatial.SetTime(1.5);
  std::printf("Part on basis::lagrange::Hexahedron<Global> proc[%d/%d] cost %f sec\n",
      i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  auto column = spatial.GetSolutionColumn();
  assert(column.size() == part.GetCellDataSize());
  spatial.SetSolutionColumn(column);
  column -= spatial.GetSolutionColumn();
  std::printf("solution.squaredNorm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.squaredNorm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  column = spatial.GetResidualColumn();
  std::printf("residual.squaredNorm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.squaredNorm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  std::printf("val_curr.squaredNorm() == %6.2e on proc[%d/%d] cost %f sec\n",
      Norm1(part), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  temporal.Update(&spatial, t_curr, dt);
  std::printf("val_next.squaredNorm() == %6.2e on proc[%d/%d] cost %f sec\n",
      Norm1(part), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  std::printf("Check the consistency between FEM and SEM implementations.\n");
  MPI_Barrier(MPI_COMM_WORLD);
  class Test : public Spatial {
    using SEM = Spatial;
    using FEM = typename Spatial::Base;

   public:
    explicit Test(Part *part_ptr)
        : Spatial(part_ptr) {
    }

    void AddFluxDivergenceOnLocalCells(Column *residual) const {
      residual->setZero();
      dynamic_cast<SEM const *>(this)->AddFluxDivergenceOnLocalCells(residual);
      *residual *= -1.0;
      dynamic_cast<FEM const *>(this)->AddFluxDivergenceOnLocalCells(residual);
    }
    void AddFluxOnLocalFaces(Column *residual) const {
      residual->setZero();
      dynamic_cast<SEM const *>(this)->AddFluxOnLocalFaces(residual);
      *residual *= -1.0;
      dynamic_cast<FEM const *>(this)->AddFluxOnLocalFaces(residual);
    }
    void AddFluxOnGhostFaces(Column *residual) const {
      residual->setZero();
      dynamic_cast<SEM const *>(this)->AddFluxOnGhostFaces(residual);
      *residual *= -1.0;
      dynamic_cast<FEM const *>(this)->AddFluxOnGhostFaces(residual);
    }
    void AddFluxOnInviscidWalls(Column *residual) const override {
      residual->setZero();
      this->SEM::AddFluxOnInviscidWalls(residual);
      *residual *= -1.0;
      this->FEM::AddFluxOnInviscidWalls(residual);
    }
    void AddFluxOnSupersonicInlets(Column *residual) const override {
      residual->setZero();
      this->SEM::AddFluxOnSupersonicInlets(residual);
      *residual *= -1.0;
      this->FEM::AddFluxOnSupersonicInlets(residual);
    }
    void AddFluxOnSupersonicOutlets(Column *residual) const override {
      residual->setZero();
      this->SEM::AddFluxOnSupersonicOutlets(residual);
      *residual *= -1.0;
      this->FEM::AddFluxOnSupersonicOutlets(residual);
    }
    void AddFluxOnSubsonicInlets(Column *residual) const override {
      residual->setZero();
      this->SEM::AddFluxOnSubsonicInlets(residual);
      *residual *= -1.0;
      this->FEM::AddFluxOnSubsonicInlets(residual);
    }
    void AddFluxOnSubsonicOutlets(Column *residual) const override {
      residual->setZero();
      this->SEM::AddFluxOnSubsonicOutlets(residual);
      *residual *= -1.0;
      this->FEM::AddFluxOnSubsonicOutlets(residual);
    }
    void AddFluxOnSmartBoundaries(Column *residual) const override {
      residual->setZero();
      this->SEM::AddFluxOnSmartBoundaries(residual);
      *residual *= -1.0;
      this->FEM::AddFluxOnSmartBoundaries(residual);
    }
  };
  auto test = Test(&part);
  test.SetSmartBoundary("4_S_27", moving);  // Top
  test.SetSmartBoundary("4_S_31", moving);  // Left
  test.SetInviscidWall("4_S_1");   // Back
  test.SetSubsonicInlet("4_S_32", moving);  // Front
  test.SetSupersonicInlet("4_S_19", moving);  // Bottom
  test.SetSubsonicOutlet("4_S_23", moving);  // Right
  test.SetSupersonicOutlet("4_S_15");  // Gap

  time_begin = MPI_Wtime();
  test.AddFluxDivergenceOnLocalCells(&column);
  std::printf("AddFluxDivergence.squaredNorm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.squaredNorm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  test.AddFluxOnLocalFaces(&column);
  std::printf("AddFluxOnLocalFaces.squaredNorm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.squaredNorm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  test.AddFluxOnGhostFaces(&column);
  std::printf("AddFluxOnGhostFaces.squaredNorm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.squaredNorm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  test.AddFluxOnInviscidWalls(&column);
  std::printf("AddFluxOnInviscidWalls.squaredNorm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.squaredNorm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  test.AddFluxOnSupersonicInlets(&column);
  std::printf("AddFluxOnSupersonicInlets.squaredNorm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.squaredNorm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  test.AddFluxOnSupersonicOutlets(&column);
  std::printf("AddFluxOnSupersonicOutlets.squaredNorm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.squaredNorm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  test.AddFluxOnSubsonicInlets(&column);
  std::printf("AddFluxOnSubsonicInlets.squaredNorm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.squaredNorm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  test.AddFluxOnSubsonicOutlets(&column);
  std::printf("AddFluxOnSubsonicOutlets.squaredNorm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.squaredNorm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);

  time_begin = MPI_Wtime();
  test.AddFluxOnSmartBoundaries(&column);
  std::printf("AddFluxOnSmartBoundaries.squaredNorm() == %6.2e on proc[%d/%d] cost %f sec\n",
      column.squaredNorm(), i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Barrier(MPI_COMM_WORLD);
}
  MPI_Finalize();
}
