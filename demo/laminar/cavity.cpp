//  Copyright 2024 PEI Weicheng
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "mpi.h"
#include "pcgnslib.h"

#include "mini/constant/index.hpp"
#include "mini/geometry/pi.hpp"
#include "mini/mesh/shuffler.hpp"
#include "mini/mesh/vtk.hpp"
#include "mini/riemann/concept.hpp"
#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/ausm.hpp"
#include "mini/riemann/euler/exact.hpp"
#include "mini/riemann/rotated/euler.hpp"
#include "mini/riemann/diffusive/navier_stokes.hpp"
#include "mini/riemann/diffusive/direct_dg.hpp"
#include "mini/polynomial/projection.hpp"
#include "mini/polynomial/hexahedron.hpp"
#include "mini/mesh/part.hpp"
#include "mini/limiter/weno.hpp"
#include "mini/temporal/rk.hpp"
#include "mini/spatial/fem.hpp"
#include "mini/spatial/dg/general.hpp"
#include "mini/spatial/dg/lobatto.hpp"
#include "mini/spatial/fr/lobatto.hpp"

#define FR

int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  int n_core, i_core;
  MPI_Comm_size(MPI_COMM_WORLD, &n_core);
  MPI_Comm_rank(MPI_COMM_WORLD, &i_core);
  cgp_mpi_comm(MPI_COMM_WORLD);

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

  auto case_name = std::string(argv[0]);
  auto pos = case_name.find_last_of('/');
  if (pos != std::string::npos) {
    case_name = case_name.substr(pos+1);
  }
  case_name.push_back('_');
  case_name += suffix;

  auto time_begin = MPI_Wtime();

  using Scalar = double;
  /* Define the Navier--Stokes equation. */
  constexpr int kDimensions = 3;
  using Gas = mini::riemann::euler::IdealGas<Scalar, 1.4>;
  using Primitive = mini::riemann::euler::Primitives<Scalar, kDimensions>;
  using Conservative = mini::riemann::euler::Conservatives<Scalar, kDimensions>;
  using Unrotated = mini::riemann::euler::Exact<Gas, kDimensions>;
  using Convection = mini::riemann::rotated::Euler<Unrotated>;
  using Diffusion = mini::riemann::diffusive::DirectDG<
      mini::riemann::diffusive::NavierStokes<Gas>
  >;
  using Riemann = mini::riemann::ConvectionDiffusion<Convection, Diffusion>;
  Riemann::SetBetaValues(2.0, 1.0 / 12);
  Riemann::SetProperty(/* nu = */0.01, /* prandtl = */0.708);

  /* Partition the mesh. */
  if (i_core == 0 && n_parts_prev != n_core) {
    using Shuffler = mini::mesh::Shuffler<idx_t, Scalar>;
    Shuffler::PartitionAndShuffle(case_name, old_file_name, n_core);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int kDegrees = 2;
  constexpr int kComponents = Riemann::kComponents;
#ifdef DGFEM
  using Projection = mini::polynomial::Projection<Scalar,
      kDimensions, kDegrees, kComponents>;
#else
  using Gx = mini::gauss::Lobatto<Scalar, kDegrees + 1>;
#endif
#ifdef DGSEM
  using Projection = mini::polynomial::Hexahedron<Gx, Gx, Gx, kComponents,
      false>;
#endif
#ifdef FR
  using Projection = mini::polynomial::Hexahedron<Gx, Gx, Gx, kComponents,
      true>;
#endif
  using Part = mini::mesh::part::Part<cgsize_t, Riemann, Projection>;
  using Cell = typename Part::Cell;
  using Face = typename Part::Face;
  using Global = typename Cell::Global;
  using Value = typename Cell::Value;
  using Coeff = typename Cell::Coeff;

  if (i_core == 0) {
    std::printf("Create %d `Part`s at %f sec\n",
        n_core, MPI_Wtime() - time_begin);
  }
  auto part = Part(case_name, i_core, n_core);
  part.SetFieldNames({"Density", "MomentumX", "MomentumY", "MomentumZ",
      "EnergyStagnationDensity"});

  /* Set initial conditions. */
  Scalar rho_given = 1.29;
  Scalar angle = 30.0;
  auto [cos, sin] = mini::geometry::CosSin(angle);
  Scalar u_given = rho_given * cos;
  Scalar v_given = rho_given * 0.0;
  Scalar w_given = rho_given * sin;
  Scalar p_given = 101325;
  auto initial_condition = [&](const Global& xyz){
    auto primitive = Primitive(rho_given, u_given, v_given, w_given, p_given);
    if (xyz[mini::constant::index::Y] < 0.9999) {
      primitive.momentum().setZero();
    }
    Value value = Gas::PrimitiveToConservative(primitive);
    return value;
  };

  if (argc == 7) {
    if (i_core == 0) {
      std::printf("[Start] Approximate() on %d cores at %f sec\n",
          n_core, MPI_Wtime() - time_begin);
    }
    for (Cell *cell_ptr : part.GetLocalCellPointers()) {
      cell_ptr->Approximate(initial_condition);
    }

    part.GatherSolutions();
    if (i_core == 0) {
      std::printf("[Start] WriteSolutions(Frame0) on %d cores at %f sec\n",
          n_core, MPI_Wtime() - time_begin);
    }
    part.WriteSolutions("Frame0");
    mini::mesh::vtk::Writer<Part>::WriteSolutions(part, "Frame0");
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

#ifdef DGFEM
  using Spatial = mini::spatial::dg::WithLimiterAndSource<Part, Limiter>;
  auto spatial = Spatial(&part, limiter);
#endif
#ifdef DGSEM
  using Spatial = mini::spatial::dg::Lobatto<Part>;
  auto spatial = Spatial(&part);
#endif
#ifdef FR
  using Spatial = mini::spatial::fr::Lobatto<Part>;
  auto spatial = Spatial(&part);
#endif

  /* Define the temporal solver. */
  constexpr int kOrders = std::min(3, kDegrees + 1);
  using Temporal = mini::temporal::RungeKutta<kOrders, Scalar>;
  auto temporal = Temporal();

  /* Set boundary conditions. */
  Scalar temperature_gradient = 0.0;
  auto moving = [&](const Global& xyz, double t){
    Value value;
    value[1] = u_given;
    value[2] = v_given;
    value[3] = w_given;
    value[4] = temperature_gradient;
    return value;
  };
  auto fixed = [&](const Global& xyz, double t){
    Value value;
    value[1] = 0;
    value[2] = 0;
    value[3] = 0;
    value[4] = temperature_gradient;
    return value;
  };
  assert(suffix == "hexa");
  if (suffix == "hexa") {
    spatial.SetSlidingWall("4_S_26", fixed);   // Left
    spatial.SetSlidingWall("4_S_18", fixed);  // Right
    spatial.SetSlidingWall("4_S_22", moving);  // Top
    spatial.SetSlidingWall("4_S_14", fixed);  // Bottom
    spatial.SetSolidWall("4_S_5");   // Back
    spatial.SetSolidWall("4_S_27");  // Front
  }

  /* Main Loop */
  auto wtime_start = MPI_Wtime();
  for (int i_step = 1; i_step <= n_steps; ++i_step) {
    double t_curr = t_start + dt * (i_step - 1);
    temporal.Update(&spatial, t_curr, dt);

    double t_next = t_curr + dt;
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
      mini::mesh::vtk::Writer<Part>::WriteSolutions(part, frame_name);
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
