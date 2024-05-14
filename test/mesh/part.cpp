//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

#include "mpi.h"
#include "pcgnslib.h"

#include "mini/mesh/cgns.hpp"
#include "mini/mesh/part.hpp"
#include "mini/mesh/vtk.hpp"
#include "mini/limiter/weno.hpp"
#include "mini/limiter/reconstruct.hpp"
#include "mini/riemann/rotated/multiple.hpp"
#include "mini/polynomial/projection.hpp"
#include "mini/polynomial/hexahedron.hpp"
#include "mini/polynomial/extrapolation.hpp"
#include "mini/input/path.hpp"  // defines INPUT_DIR

#include "test/mesh/part.hpp"

template <class Part>
void Process(Part *part_ptr, const std::string &solution_name) {
  auto quadrangle = mini::coordinate::Quadrangle4<Scalar, kDimensions>();
  part_ptr->InstallPrototype(4,
      std::make_unique<QuadrangleIntegrator>(quadrangle));
  auto hexahedron = mini::coordinate::Hexahedron8<Scalar>();
  part_ptr->InstallPrototype(8,
      std::make_unique<HexahedronIntegrator>(hexahedron));
  part_ptr->ReadCgnsFile();
  part_ptr->SetFieldNames({"U1", "U2"});
  double volume = 0.0, area = 0.0;
  int n_cells = 0, n_faces = 0;
  for (const auto &cell : part_ptr->GetLocalCells()) {
    volume += cell.volume();
    n_cells += 1;
    n_faces += cell.adj_faces_.size();
    for (auto *face_ptr : cell.adj_faces_) {
      assert(face_ptr);
      area += face_ptr->area();
    }
    assert(part_ptr->GetCellDataOffset(cell.id()) == cell.id() * cell.K * cell.N);
  }
  using Cell = typename Part::Cell;
  assert(part_ptr->GetCellDataSize() == n_cells * Cell::K * Cell::N);
  std::printf("On proc[%d/%d], avg_volume = %f = %f / %d\n",
      i_core, n_core, volume / n_cells, volume, n_cells);
  std::printf("On proc[%d/%d], avg_area = %f = %f / %d\n",
      i_core, n_core, area / n_faces, area, n_faces);
  std::printf("Run Approximate() on proc[%d/%d] at %f sec\n",
      i_core, n_core, MPI_Wtime() - time_begin);
  for (auto *cell_ptr : part_ptr->GetLocalCellPointers()) {
    cell_ptr->Approximate(func);
  }
  std::printf("Run Reconstruct() on proc[%d/%d] at %f sec\n",
      i_core, n_core, MPI_Wtime() - time_begin);
  using Cell = typename Part::Cell;
  auto lazy_limiter = mini::limiter::weno::Lazy<Cell>(
      /* w0 = */0.001, /* eps = */1e-6, /* verbose = */false);
  mini::limiter::Reconstruct(part_ptr, lazy_limiter);
  std::printf("Run Write() on proc[%d/%d] at %f sec\n",
      i_core, n_core, MPI_Wtime() - time_begin);
  part_ptr->GatherSolutions();
  part_ptr->WriteSolutions(solution_name);
  mini::mesh::vtk::Writer<Part>::WriteSolutions(*part_ptr, solution_name);
}

// mpirun -n 4 ./part [<case_name> [<input_dir>]]]
int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &n_core);
  MPI_Comm_rank(MPI_COMM_WORLD, &i_core);
  cgp_mpi_comm(MPI_COMM_WORLD);

  auto case_name = std::string("double_mach");
  if (argc > 1)
    case_name = argv[1];
  auto input_dir =  std::string(INPUT_DIR);
  if (argc > 2)
    input_dir = argv[2];

  time_begin = MPI_Wtime();

  // create the cgns and partition files to be processed
  if (i_core == 0) {
    std::printf("Run `./shuffler %d %s %s` on proc[%d/%d] at %f sec\n",
        n_core, case_name.c_str(), input_dir.c_str(),
        i_core, n_core, MPI_Wtime() - time_begin);
    auto cmd = "./shuffler " + std::to_string(n_core) + ' ' + case_name + ' '
        + input_dir;
    if (std::system(cmd.c_str()))
      throw std::runtime_error(cmd + std::string(" failed."));
  }
  MPI_Barrier(MPI_COMM_WORLD);

  /* aproximated by Projection on OrthoNormal basis */
{
  std::printf("Run Part() on proc[%d/%d] at %f sec\n",
      i_core, n_core, MPI_Wtime() - time_begin);
  using Projection = mini::polynomial::Projection<
      Scalar, kDimensions, kDegrees, kComponents>;
  using Part = mini::mesh::part::Part<cgsize_t, Riemann, Projection>;
  auto part = Part(case_name, i_core, n_core);
  Process(&part, "Projection");
}
  /* aproximated by Interpolation on Lagrange basis */
{
  std::printf("Run Part() on proc[%d/%d] at %f sec\n",
      i_core, n_core, MPI_Wtime() - time_begin);
  using Interpolation = mini::polynomial::Hexahedron<Gx, Gx, Gx, kComponents, true>;
  using Extrapolation = mini::polynomial::Extrapolation<Interpolation>;
  using Part = mini::mesh::part::Part<cgsize_t, Riemann, Extrapolation>;
  auto part = Part(case_name, i_core, n_core);
  Process(&part, "Interpolation");
}
  std::printf("Run MPI_Finalize() on proc[%d/%d] at %f sec\n",
      i_core, n_core, MPI_Wtime() - time_begin);
  MPI_Finalize();
}
