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
using Solution = mini::mesh::cgns::Solution<double>;
using Field = std::vector<double>;

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
    auto &section = zone.GetSection(1);
    if (section.type() == CGNS_ENUMV(HEXA_8)) {
      using Coordinate = mini::coordinate::Hexahedron8<double>;
      using Global = typename Coordinate::Global;
      using Gx = mini::integrator::Lobatto<double, 3>;
      using Gy = mini::integrator::Lobatto<double, 3>;
      using Gz = mini::integrator::Lobatto<double, 3>;
      using Integrator = mini::integrator::Hexahedron<Gx, Gy, Gz>;
      for (int i_cell = section.CellIdMin(); i_cell <= section.CellIdMin(); ++i_cell) {
        auto *nodes = section.GetNodeIdList(i_cell);
        auto coordinate = Coordinate{
          Global(x[nodes[0]], y[nodes[0]], z[nodes[0]]),
          Global(x[nodes[1]], y[nodes[1]], z[nodes[1]]),
          Global(x[nodes[2]], y[nodes[2]], z[nodes[2]]),
          Global(x[nodes[3]], y[nodes[3]], z[nodes[3]]),
          Global(x[nodes[4]], y[nodes[4]], z[nodes[4]]),
          Global(x[nodes[5]], y[nodes[5]], z[nodes[5]]),
          Global(x[nodes[6]], y[nodes[6]], z[nodes[6]]),
          Global(x[nodes[7]], y[nodes[7]], z[nodes[7]]) };
        auto integrator = Integrator(coordinate);
        double min_jacobian = DBL_MAX, max_jacobian = -DBL_MAX, volume = 0.0;
        for (int q = 0; q < Integrator::Q; q++) {
          double jacobian = integrator.GetJacobianDeterminant(q);
          min_jacobian = std::min(min_jacobian, jacobian);
          max_jacobian = std::min(max_jacobian, jacobian);
          volume += jacobian;
        }
        min_jacobians.at(i_cell) = min_jacobian;
        max_jacobians.at(i_cell) = max_jacobian;
        volumes.at(i_cell) = volume;
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
