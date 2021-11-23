//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "mpi.h"
#include "pcgnslib.h"

#include "mini/mesh/cgns.hpp"
#include "mini/mesh/part.hpp"
#include "mini/polynomial/limiter.hpp"

// mpirun -n 4 ./part
int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);
  int comm_size, comm_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  cgp_mpi_comm(MPI_COMM_WORLD);

  auto time_begin = MPI_Wtime();
  if (comm_rank == 0) {
    std::printf("Run ./shuffler on proc[%d/%d] at %f sec\n",
        comm_rank, comm_size, MPI_Wtime() - time_begin);
    auto cmd = std::string("./shuffler " + std::to_string(comm_size));
    std::system(cmd.c_str());
  }
  MPI_Barrier(MPI_COMM_WORLD);

  std::printf("Run Part() on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  auto part = mini::mesh::cgns::Part<cgsize_t, double, 2>(
      "double_mach_hexa", comm_rank);
  double volume = 0.0, area = 0.0;
  int n_cells = 0, n_faces = 0;
  const auto *part_ptr = &part;
  part_ptr->ForEachLocalCell([&](const auto &cell){
    volume += cell.volume();
    n_cells += 1;
    n_faces += cell.adj_faces_.size();
    for (auto* face_ptr : cell.adj_faces_) {
      assert(face_ptr);
      area += face_ptr->area();
    }
  });
  std::printf("On proc[%d/%d], avg_volume = %f = %f / %d\n",
      comm_rank, comm_size, volume / n_cells, volume, n_cells);
  std::printf("On proc[%d/%d], avg_area = %f = %f / %d\n",
      comm_rank, comm_size, area / n_faces, area, n_faces);
  std::printf("Run Project() on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  part.Project([](auto const& xyz){
    auto r = std::hypot(xyz[0] - 2, xyz[1] - 0.5);
    mini::algebra::Matrix<double, 2, 1> col;
    col[0] = r;
    col[1] = 1 - r + (r >= 1);
    return col;
  });
  std::printf("Run Reconstruct() on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  using MyCell = mini::mesh::cgns::Cell<cgsize_t, double, 2>;
  auto lazy_limiter = mini::polynomial::LazyWeno<MyCell>(
      /* w0 = */0.001, /* eps = */1e-6, /* verbose = */false);
  part.Reconstruct(lazy_limiter);
  std::printf("Run Write() on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  part.GatherSolutions();
  part.WriteSolutions();
  // part.WriteSolutionsOnGaussPoints();
  part.WriteSolutionsOnCellCenters();
  std::printf("Run MPI_Finalize() on proc[%d/%d] at %f sec\n",
      comm_rank, comm_size, MPI_Wtime() - time_begin);
  MPI_Finalize();
}