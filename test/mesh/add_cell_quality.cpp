// Copyright 2024 PEI Weicheng

#include <cstdio>
#include <string>
#include <vector>

#include "cgnslib.h"
#include "gtest/gtest.h"

#include "mini/mesh/cgns.hpp"
#include "mini/coordinate/hexahedron.hpp"
#include "mini/integrator/hexahedron.hpp"
#include "mini/integrator/lobatto.hpp"

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
  if (file.CountBases() != 1) {
    throw std::runtime_error("Only 1-CGNSBase_t file is supported!");
  }
  auto &base = file.GetBase(1);

  // read the unique Zone_t
  if (base.CountZones() != 1) {
    throw std::runtime_error("Only 1-Zone_t CGNSBase_t is supported!");
  }
  auto &zone = base.GetZone(1);
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
  auto &min_jacobians = cell_data->AddField("MinDetJac");
  auto &max_jacobians = cell_data->AddField("MaxDetJac");
  auto &volumes = cell_data->AddField("Volume");

  // read Elements_t's
  auto n_sect = zone.CountSections();
  for (int i_sect = 1; i_sect <= n_sect; ++i_sect) {
    auto &section = zone.GetSection(i_sect);
    if (section.type() == CGNS_ENUMV(HEXA_8)) {
      for (int i_cell = section.CellIdMin(); i_cell <= section.CellIdMax(); ++i_cell) {
        auto const *nodes = section.GetNodeIdList(i_cell);
        auto coordinate = Coordinate{
            GetGlobal(coordinates, nodes[0]), GetGlobal(coordinates, nodes[1]),
            GetGlobal(coordinates, nodes[2]), GetGlobal(coordinates, nodes[3]),
            GetGlobal(coordinates, nodes[4]), GetGlobal(coordinates, nodes[5]),
            GetGlobal(coordinates, nodes[6]), GetGlobal(coordinates, nodes[7]) };
            // Global(x[nodes[7] - 1], y[nodes[7] - 1], z[nodes[7] - 1]) };
        auto const &integrator = Integrator(coordinate);
        double min_jacobian = DBL_MAX, max_jacobian = -DBL_MAX;
        for (int q = 0; q < Integrator::Q; q++) {
          double jacobian = integrator.GetJacobianDeterminant(q);
          min_jacobian = std::min(min_jacobian, jacobian);
          max_jacobian = std::max(max_jacobian, jacobian);
        }
        min_jacobians.at(i_cell) = min_jacobian;
        max_jacobians.at(i_cell) = max_jacobian;
        volumes.at(i_cell) = integrator.volume();
      }
    }
  }

  // write the augmented file
  for (int i = 0; i < 5; ++i) {
    cgns_file.pop_back();
  }
  auto output = cgns_file + "_with_quality.cgns";
  std::printf("writing to %s ...\n", output.c_str());
  file.Write(output);
}
