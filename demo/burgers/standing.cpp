//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "mpi.h"
#include "pcgnslib.h"

#include "mini/mesh/shuffler.hpp"
#include "mini/mesh/vtk.hpp"
#include "mini/temporal/rk.hpp"
#include "mini/coordinate/quadrangle.hpp"
#include "mini/integrator/quadrangle.hpp"
#include "mini/coordinate/hexahedron.hpp"
#include "mini/integrator/hexahedron.hpp"

using Scalar = double;
/* Define the Burgers equation. */
constexpr int kComponents = 1;
constexpr int kDimensions = 3;
constexpr int kDegrees = 2;

#include "mini/riemann/rotated/burgers.hpp"
using Riemann = mini::riemann::rotated::Burgers<Scalar, kDimensions>;

#define FR // exactly one of (DGFEM, DGSEM, FR) must be defined

#ifdef DGFEM
#include "mini/integrator/legendre.hpp"
using Gx = mini::integrator::Legendre<Scalar, kDegrees + 1>;
#include "mini/polynomial/projection.hpp"
using Polynomial = mini::polynomial::Projection<Scalar, kDimensions, kDegrees, kComponents>;

#else  // common for DGSEM and FR
#include "mini/integrator/lobatto.hpp"
using Gx = mini::integrator::Lobatto<Scalar, kDegrees + 1>;

#include "mini/polynomial/hexahedron.hpp"
#include "mini/polynomial/extrapolation.hpp"
using Interpolation = mini::polynomial::Hexahedron<Gx, Gx, Gx, kComponents, true>;
using Polynomial = mini::polynomial::Extrapolation<Interpolation>;
#endif

#include "mini/mesh/part.hpp"
using Part = mini::mesh::part::Part<cgsize_t, Polynomial>;
using Cell = typename Part::Cell;
using Face = typename Part::Face;
using Global = typename Cell::Global;
using Value = typename Cell::Value;
using Coeff = typename Cell::Coeff;

static void InstallIntegratorPrototypes(Part *part_ptr) {
  auto quadrangle = mini::coordinate::Quadrangle4<Scalar, kDimensions>();
  using QuadrangleIntegrator
    = mini::integrator::Quadrangle<kDimensions, Gx, Gx>;
  part_ptr->InstallPrototype(4,
      std::make_unique<QuadrangleIntegrator>(quadrangle));
  auto hexahedron = mini::coordinate::Hexahedron8<Scalar>();
  using HexahedronIntegrator
      = mini::integrator::Hexahedron<Gx, Gx, Gx>;
  part_ptr->InstallPrototype(8,
      std::make_unique<HexahedronIntegrator>(hexahedron));
  part_ptr->BuildGeometry();
}

using VtkWriter = mini::mesh::vtk::Writer<Part>;

#define VISCOSITY  // one of (LIMITER, VISCOSITY) must be defined

#ifdef LIMITER

#ifdef DGFEM
#include "mini/spatial/dg/general.hpp"
using General = mini::spatial::dg::General<Part, Riemann>;

#elif defined(DGSEM)
#include "mini/spatial/dg/lobatto.hpp"
using General = mini::spatial::dg::Lobatto<Part, Riemann>;

#elif defined(FR)
#include "mini/spatial/fr/lobatto.hpp"
using General = mini::spatial::fr::Lobatto<Part, Riemann>;
#endif

#include "mini/limiter/weno.hpp"
#include "mini/spatial/with_limiter.hpp"
using Limiter = mini::limiter::weno::Lazy<Cell>;
using Spatial = mini::spatial::WithLimiter<General, Limiter>;

#endif

#ifdef VISCOSITY

#include "mini/riemann/concept.hpp"
#include "mini/riemann/diffusive/linear.hpp"
#include "mini/riemann/diffusive/direct_dg.hpp"
#include "mini/spatial/viscosity.hpp"
#include "mini/spatial/with_viscosity.hpp"

using Diffusion = mini::riemann::diffusive::DirectDG<
    mini::riemann::diffusive::Isotropic<Scalar, kComponents>>;
using RiemannWithViscosity = mini::spatial::EnergyBasedViscosity<Part,
    mini::riemann::ConvectionDiffusion<Riemann, Diffusion>>;
static_assert(mini::riemann::ConvectiveDiffusive<RiemannWithViscosity>);

#if defined(FR)
#include "mini/spatial/fr/lobatto.hpp"
using General = mini::spatial::fr::Lobatto<Part, RiemannWithViscosity>;
#endif

using Spatial = mini::spatial::WithViscosity<General>;

#endif

int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  int n_core, i_core;
  MPI_Comm_size(MPI_COMM_WORLD, &n_core);
  MPI_Comm_rank(MPI_COMM_WORLD, &i_core);
  cgp_mpi_comm(MPI_COMM_WORLD);

  Riemann::Convection::SetJacobians(1, 0, 0);
#ifdef VISCOSITY
  Diffusion::SetProperty(0.0);
  Diffusion::SetBetaValues(2.0, 1.0 / 12);
#endif
  if (argc < 7) {
    if (i_core == 0) {
      std::cout << "usage:\n"
          << "  mpirun -n <n_core> " << argv[0] << " <cgns_file> <hexa|tetra>"
          << " <t_start> <t_stop> <n_steps_per_frame> <n_frames>"
          << " [<i_frame_start> [n_parts_prev]]\n";
    }
    MPI_Finalize();
    exit(0);
  }
  auto old_file_name = std::string(argv[1]);
  auto suffix = std::string(argv[2]);
  double t_start = std::atof(argv[3]);
  double t_stop = std::atof(argv[4]);
  int n_steps_per_frame = std::atoi(argv[5]);
  int n_frames = std::atoi(argv[6]);
  int n_steps = n_frames * n_steps_per_frame;
  auto dt = (t_stop - t_start) / n_steps;
  int i_frame = 0;
  if (argc > 7) {
    i_frame = std::atoi(argv[7]);
  }
  int n_parts_prev = n_core;
  if (argc > 8) {
    n_parts_prev = std::atoi(argv[8]);
  }

  std::string case_name = "standing_" + suffix;

  auto time_begin = MPI_Wtime();

  /* Partition the mesh. */
  if (i_core == 0 && (argc == 7 || n_parts_prev != n_core)) {
    using Shuffler = mini::mesh::Shuffler<idx_t, Scalar>;
    Shuffler::PartitionAndShuffle(case_name, old_file_name, n_core);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if (i_core == 0) {
    std::printf("Create %d `Part`s at %f sec\n",
        n_core, MPI_Wtime() - time_begin);
  }
  auto part = Part(case_name, i_core, n_core);
  InstallIntegratorPrototypes(&part);
  part.SetFieldNames({"U"});

  /* Build a `Spatial` object. */
#ifdef LIMITER
  auto limiter = Limiter(/* w0 = */0.001, /* eps = */1e-6);
  auto spatial = Spatial(&limiter, &part);
#else
  auto spatial = Spatial(&part);
  RiemannWithViscosity::SetTimeScale(1.0);
  VtkWriter::AddExtraField("CellViscosity", [&](Cell const &cell, Global const &global, Value const &value){
    return RiemannWithViscosity::GetPropertyOnCell(cell.id(), 0)[0];
  });
#endif

  /* Set initial conditions. */
  auto initial_condition = [&](const Global& xyz){
    auto x = xyz[0];
    Value val;
    val[0] = x * (x - 2.0) * (x - 4.0);
    return val;
  };

  if (argc == 7) {
    if (i_core == 0) {
      std::printf("[Start] Approximate() on %d cores at %f sec\n",
          n_core, MPI_Wtime() - time_begin);
    }
    for (Cell *cell_ptr : part.GetLocalCellPointers()) {
      cell_ptr->Approximate(initial_condition);
    }
    part.ShareGhostCellCoeffs();
    part.UpdateGhostCellCoeffs();
    part.GatherSolutions();
#ifdef VISCOSITY
    RiemannWithViscosity::Viscosity::UpdateProperties();
#endif
    if (i_core == 0) {
      std::printf("[Start] WriteSolutions(Frame0) on %d cores at %f sec\n",
          n_core, MPI_Wtime() - time_begin);
    }
    part.WriteSolutions("Frame0");
    VtkWriter::WriteSolutions(part, "Frame0");
  } else {
    if (i_core == 0) {
      std::printf("[Start] ReadSolutions(Frame%d) on %d cores at %f sec\n",
          i_frame, n_core, MPI_Wtime() - time_begin);
    }
    std::string soln_name = (n_parts_prev != n_core)
        ? "shuffled" : "Frame" + std::to_string(i_frame);
    part.ReadSolutions(soln_name);
    part.ScatterSolutions();
  }

  /* Define the temporal solver. */
  constexpr int kOrders = std::min(3, kDegrees + 1);
  using Temporal = mini::temporal::RungeKutta<kOrders, Scalar>;
  auto temporal = Temporal();

  /* Set boundary conditions. */
  if (suffix == "tetra") {
    spatial.SetInviscidWall("3_S_27");  // Top
    spatial.SetInviscidWall("3_S_31");  // Left
    spatial.SetInviscidWall("3_S_1");   // Back
    spatial.SetInviscidWall("3_S_32");  // Front
    spatial.SetInviscidWall("3_S_19");  // Bottom
    spatial.SetInviscidWall("3_S_23");  // Right
    spatial.SetInviscidWall("3_S_15");  // Gap
  } else {
    assert(suffix == "hexa");
    spatial.SetInviscidWall("4_S_27");  // Top
    spatial.SetInviscidWall("4_S_31");  // Left
    spatial.SetInviscidWall("4_S_1");   // Back
    spatial.SetInviscidWall("4_S_32");  // Front
    spatial.SetInviscidWall("4_S_19");  // Bottom
    spatial.SetInviscidWall("4_S_23");  // Right
    spatial.SetInviscidWall("4_S_15");  // Gap
  }

  /* Main Loop */
  auto wtime_start = MPI_Wtime();
  for (int i_step = 1; i_step <= n_steps; ++i_step) {
    double t_curr = t_start + dt * (i_step - 1);
    temporal.Update(&spatial, t_curr, dt);

    auto wtime_curr = MPI_Wtime() - wtime_start;
    auto wtime_total = wtime_curr * n_steps / i_step;
    if (i_core == 0) {
      std::printf("[Done] Update(Step%d/%d) on %d cores at %f / %f sec\n",
          i_step, n_steps, n_core, wtime_curr, wtime_total);
    }

    if (i_step % n_steps_per_frame == 0) {
      ++i_frame;
      part.GatherSolutions();
      if (i_core == 0) {
        std::printf("[Start] WriteSolutions(Frame%d) on %d cores at %f sec\n",
            i_frame, n_core, MPI_Wtime() - wtime_start);
      }
      auto frame_name = "Frame" + std::to_string(i_frame);
      part.WriteSolutions(frame_name);
      VtkWriter::WriteSolutions(part, frame_name);
    }
  }

  if (i_core == 0) {
    std::printf("time-range = [%f, %f], frame-range = [%d, %d], dt = %f\n",
        t_start, t_stop, i_frame - n_frames, i_frame, dt);
    std::printf("[Start] MPI_Finalize() on %d cores at %f sec\n",
        n_core, MPI_Wtime() - time_begin);
  }
  MPI_Finalize();
}
