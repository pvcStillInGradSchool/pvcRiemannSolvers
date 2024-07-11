//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "mpi.h"
#include "pcgnslib.h"

#include "mini/mesh/shuffler.hpp"

#include "sourceless.hpp"

#include "mini/coordinate/quadrangle.hpp"
#include "mini/integrator/quadrangle.hpp"
#include "mini/coordinate/hexahedron.hpp"
#include "mini/integrator/hexahedron.hpp"

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
#ifdef DGFEM  // TODO(PVC): install prototypes for Triangle, Tetrahedron, etc.
#endif
  part_ptr->BuildGeometry();
}

#include "mini/mesh/vtk.hpp"
using VtkWriter = mini::mesh::vtk::Writer<Part>;

#ifdef VISCOSITY
#include "mini/limiter/average.hpp"
#include "mini/limiter/majority.hpp"
#endif

#include <fstream>
#include <nlohmann/json.hpp>

int Main(int argc, char* argv[], IC ic, BC bc) {
  MPI_Init(NULL, NULL);
  int n_core, i_core;
  MPI_Comm_size(MPI_COMM_WORLD, &n_core);
  MPI_Comm_rank(MPI_COMM_WORLD, &i_core);
  cgp_mpi_comm(MPI_COMM_WORLD);

  if (argc != 2) {
    if (i_core == 0) {
      std::cout << "usage:\n"
          << "  mpirun -n <n_core> " << argv[0] << " <json_input_file>\n";
    }
    MPI_Finalize();
    exit(0);
  }

  auto json_input_file = std::ifstream(argv[1]);
  auto json_object = nlohmann::json::parse(json_input_file);

  std::string old_file_name = json_object.at("cgns_file");
  std::string suffix = json_object.at("cell_type");
  const double t_start = json_object.at("t_start");
  const double t_stop = json_object.at("t_stop");
  const int n_frames = json_object.at("n_frames");
  const double dt_per_frame = (t_stop - t_start) / n_frames;
  const int n_steps_per_frame = json_object.at("n_steps_per_frame");
  const int n_steps = n_frames * n_steps_per_frame;
  const double dt_max = (t_stop - t_start) / n_steps;
  const int i_frame_prev = json_object.at("i_frame_prev");
  // `i_frame_prev` might be -1, which means no previous result to be loaded.
  const int i_frame_min = std::max(i_frame_prev, 0);
  const int i_frame_max = i_frame_min + n_frames;
  int n_parts_prev = n_core;
  if (i_frame_prev >= 0) {
    n_parts_prev = json_object.at("n_parts_prev");
  }
  std::string case_name = json_object.at("problem_name");
  (case_name += "_h=") += json_object.at("cell_length");
  (case_name += "_p=") += std::to_string(kDegrees);
#ifdef LIMITER
  case_name += "_limiter";
#else
  case_name += "_viscosity";
#endif
  case_name.push_back('_');
  case_name += suffix;

  auto time_begin = MPI_Wtime();

  /* Partition the mesh. */
  if (i_core == 0 && (i_frame_prev < 0 || n_parts_prev != n_core)) {
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
  part.SetFieldNames({"Density", "MomentumX", "MomentumY", "MomentumZ",
      "EnergyStagnationDensity"});

#ifdef LIMITER
  /* Build a `Limiter` object. */
  auto limiter = Limiter(/* w0 = */0.001, /* eps = */1e-6);
  auto spatial = Spatial(&limiter, &part);
#else
  Diffusion::SetProperty(0.0);
  Diffusion::SetBetaValues(
      json_object.at("ddg_beta_0"), json_object.at("ddg_beta_1"));
  auto spatial = Spatial(&part);
  RiemannWithViscosity::SetTimeScale(json_object.at("time_scale"));
  for (int k = 0; k < kComponents; ++k) {
    VtkWriter::AddCellData("CellViscosity" + std::to_string(k + 1),
        [k](Cell const &cell) {
            return RiemannWithViscosity::GetPropertyOnCell(cell.id(), 0)[k]; });
  }
#endif

  /* Initialization. */
  if (i_frame_prev < 0) {
    spatial.Approximate(ic);
    if (i_core == 0) {
      std::printf("[Done] Approximate() on %d cores at %f sec\n",
          n_core, MPI_Wtime() - time_begin);
    }

#ifdef LIMITER
    mini::limiter::Reconstruct(&part, &limiter);
    if (suffix == "tetra") {
      mini::limiter::Reconstruct(&part, &limiter);
    }
#else  // VISCOSITY
    std::string initial_limiter = json_object.at("initial_limiter");
    if (initial_limiter == "majority") {
      mini::limiter::majority::Reconstruct(spatial.part_ptr());
    } else {
      assert(initial_limiter == "average");
      mini::limiter::average::Reconstruct(spatial.part_ptr());
    }
    RiemannWithViscosity::Viscosity::UpdateProperties();
#endif
    if (i_core == 0) {
      std::printf("[Done] Reconstruct() on %d cores at %f sec\n",
          n_core, MPI_Wtime() - time_begin);
    }

    part.GatherSolutions();
    part.WriteSolutions("Frame0");
    VtkWriter::WriteSolutions(part, "Frame0");
    if (i_core == 0) {
      std::printf("[Done] WriteSolutions(Frame0) on %d cores at %f sec\n",
          n_core, MPI_Wtime() - time_begin);
    }
  } else {
    std::string soln_name = (n_parts_prev != n_core)
        ? "shuffled" : "Frame" + std::to_string(i_frame_min);
    part.ReadSolutions(soln_name);
    part.ScatterSolutions();
    if (i_core == 0) {
      std::printf("[Done] ReadSolutions(Frame%d) on %d cores at %f sec\n",
          i_frame_min, n_core, MPI_Wtime() - time_begin);
    }
  }

  /* Define the temporal solver. */
  auto temporal = Temporal();

  /* Set boundary conditions. */
  bc(suffix, &spatial);

  /* Main Loop */
  auto wtime_start = MPI_Wtime();
  double t_curr = t_start;
  for (int i_frame = i_frame_min; i_frame < i_frame_max; ++i_frame) {
    double t_next = t_curr + dt_per_frame;
    while (t_curr < t_next) {
      double dt_guess = std::min(t_next - t_curr, dt_max);
      double dt_local = (&spatial)->GetTimeStep(dt_guess, kOrders);
      assert(dt_local <= dt_guess);
      double dt;  // i.e. dt_global
      MPI_Allreduce(&dt_local, &dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      if (dt_local <= dt) {
        assert(dt_local == dt);
        if (dt < dt_guess) {
          std::printf("[Next] dt = %4.2e determined by dt_local on core[%d]\n",
              dt, i_core);
        } else if (i_core == 0) {
          std::printf("[Next] dt = %4.2e determined by %s\n",
              dt, dt < dt_max ? "t_next - t_curr" : "dt_max");
        }
      }
      temporal.Update(&spatial, t_curr, dt);
      t_curr += dt;
      // Print current percentage:
      double wtime_curr = MPI_Wtime() - wtime_start;
      double percentage = (t_curr - t_start) / (t_stop - t_start);
      double wtime_total = wtime_curr / percentage;
      if (i_core == 0) {
        std::printf("[Done] %4.2f / 100 on %d cores at %f / %f sec\n",
            percentage * 100, n_core, wtime_curr, wtime_total);
      }
    }

    // Write the solutions at the next frame:
    auto frame_name = "Frame" + std::to_string(i_frame + 1);
    part.GatherSolutions();
    part.WriteSolutions(frame_name);
    VtkWriter::WriteSolutions(part, frame_name);
    if (i_core == 0) {
      std::printf("[Done] WriteSolutions(Frame%d) on %d cores at %f sec\n",
          i_frame + 1, n_core, MPI_Wtime() - wtime_start);
    }
  }

  if (i_core == 0) {
    json_object["n_parts_curr"] = n_core;
    std::string output_name = case_name;
    output_name += "/Frame";
    output_name += std::to_string(i_frame_min) + "to";
    output_name += std::to_string(i_frame_max) + ".json";
    auto json_output_file = std::ofstream(output_name);
    json_output_file << std::setw(2) << json_object << std::endl;
    std::printf("time-range = [%f, %f], frame-range = [%d, %d]\n",
        t_start, t_stop, i_frame_min, i_frame_max);
    std::printf("[Start] MPI_Finalize() on %d cores at %f sec\n",
        n_core, MPI_Wtime() - time_begin);
  }
  MPI_Finalize();
  return 0;
}
