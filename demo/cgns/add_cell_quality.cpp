// Copyright 2024 PEI Weicheng

#include <cstdio>
#include <algorithm>
#include <chrono>
#include <limits>
#include <string>
#include <vector>

#include "cgnslib.h"
#include "omp.h"

#include "mini/mesh/cgns.hpp"
#include "mini/coordinate/hexahedron.hpp"
#include "mini/integrator/hexahedron.hpp"
#include "mini/integrator/lobatto.hpp"

using Clock = std::chrono::high_resolution_clock;

using mini::mesh::cgns::GridLocation;
using mini::mesh::cgns::ElementType;
using mini::mesh::cgns::DataType;

using File = mini::mesh::cgns::File<double>;
using Coordinates = mini::mesh::cgns::Coordinates<double>;
using Solution = mini::mesh::cgns::Solution<double>;

using Coordinate = mini::coordinate::Hexahedron8<double>;
using Global = typename Coordinate::Global;
using Gx = mini::integrator::Lobatto<double, 3>;
using Integrator = mini::integrator::Hexahedron<Gx, Gx, Gx>;

inline Global GetGlobal(Coordinates const &coordinates, cgsize_t i_node/* 1-based */) {
  return { coordinates.x(i_node), coordinates.y(i_node), coordinates.z(i_node) };
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::printf("usage:\n  %s <cgns_file> <n_thread>\n", argv[0]);
    return -1;
  }
  auto cgns_file = std::string(argv[1]);
  int n_thread = std::atoi(argv[2]);

  auto file = File(cgns_file);

  // read the unique CGNSBase_t
  file.ReadBases();
  auto &base = file.GetUniqueBase();

  // read the unique Zone_t
  auto &zone = base.GetUniqueZone();
  auto n_node = zone.CountNodes();
  auto n_cell = zone.CountCells();
  std::printf("n_node = %d, n_cell = %d\n", n_node, n_cell);

  // read GridCoodinates_t
  auto &coordinates = zone.GetCoordinates();
  auto &x = coordinates.x();
  auto &y = coordinates.y();
  auto &z = coordinates.z();

  // init the quality fields
  int n_sol = zone.CountSolutions();
  Solution *cell_data = nullptr;
  for (int i_sol = 1; i_sol <= n_sol; i_sol++) {
    auto &solution = zone.GetSolution(i_sol);
    if (solution.localtion() == CGNS_ENUMV(CellCenter)) {
      cell_data = &solution;
    }
  }
  if (!cell_data) {
    // create a new one
    auto &solution = zone.AddSolution("CellQuality", CGNS_ENUMV(CellCenter));
    cell_data = &solution;
  }
  // add the fields
  auto &volumes = cell_data->AddField("Volume");
  auto &det_jac_mins = cell_data->AddField("DetJacMin");
  auto &det_jac_ratios = cell_data->AddField("DetJacRatio");

  // read Elements_t's
  auto n_sect = zone.CountSections();

  for (int i_sect = 1; i_sect <= n_sect; ++i_sect) {
    auto const &section = zone.GetSection(i_sect);
    if (section.type() == CGNS_ENUMV(HEXA_8)) {
      auto start = Clock::now();
      int i_cell_max = section.CellIdMax();
#     pragma omp parallel for num_threads(n_thread)
      for (int i_cell = section.CellIdMin(); i_cell <= i_cell_max; ++i_cell) {
        auto const *nodes = section.GetNodeIdList(i_cell);
        auto coordinate = Coordinate{
            GetGlobal(coordinates, nodes[0]), GetGlobal(coordinates, nodes[1]),
            GetGlobal(coordinates, nodes[2]), GetGlobal(coordinates, nodes[3]),
            GetGlobal(coordinates, nodes[4]), GetGlobal(coordinates, nodes[5]),
            GetGlobal(coordinates, nodes[6]), GetGlobal(coordinates, nodes[7]) };
            // Global(x[nodes[7] - 1], y[nodes[7] - 1], z[nodes[7] - 1]) };
        auto const &integrator = Integrator(coordinate);
        double det_jac_min = std::numeric_limits<double>::max();
        double det_jac_max = std::numeric_limits<double>::lowest();
        for (int q = 0; q < Integrator::Q; q++) {
          double det_jac = integrator.GetJacobianDeterminant(q);
          det_jac_min = std::min(det_jac_min, det_jac);
          det_jac_max = std::max(det_jac_max, det_jac);
        }
        volumes.at(i_cell) = integrator.volume();
        det_jac_mins.at(i_cell) = det_jac_min;
        det_jac_ratios.at(i_cell) = det_jac_min / det_jac_max;
        // std::printf("%d / %d\n", i_cell, n_cell);
      }
      auto stop = Clock::now();
      auto cost = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
      std::printf("Adding cell quality to section %s by %d threads costs %ld seconds\n",
          section.name().c_str(), n_thread, cost.count());
    }
  }
  auto is_negative = [](double value) { return value < 0; };
  std::printf("%ld cells have volume < 0\n",
      std::ranges::count_if(volumes, is_negative));
  std::printf("%ld cells have min(det(J)) < 0\n",
      std::ranges::count_if(det_jac_mins, is_negative));
  std::printf("%ld cells have min(det(J)) < max(det(J)) * 0.01\n",
      std::ranges::count_if(det_jac_ratios,
          [](double value) { return value < 0.01; }));

  // write the augmented file
  for (int i = 0; i < 5; ++i) {
    cgns_file.pop_back();
  }
  auto output = cgns_file + "_with_quality.cgns";
  std::printf("writing to %s ...\n", output.c_str());
  file.Write(output);
}
