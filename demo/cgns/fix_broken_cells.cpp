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
using Zone = mini::mesh::cgns::Zone<double>;
using Coordinates = mini::mesh::cgns::Coordinates<double>;
using Solution = mini::mesh::cgns::Solution<double>;

using Coordinate = mini::coordinate::Hexahedron8<double>;
using Global = typename Coordinate::Global;
using Gx = mini::integrator::Lobatto<double, 3>;
using Integrator = mini::integrator::Hexahedron<Gx, Gx, Gx>;

inline Global GetGlobal(Coordinates const &coordinates, cgsize_t i_node/* 1-based */) {
  return { coordinates.x(i_node), coordinates.y(i_node), coordinates.z(i_node) };
}

inline bool Invalid(double det_jac_min, double det_jac_max) {
  return det_jac_min <= 0 || det_jac_min < det_jac_max * 0.01;
}

bool FindBrokenPoints(Zone const &zone, Coordinates const &coordinates, int n_thread,
    std::unordered_set<cgsize_t> *broken_points) {
  auto start = Clock::now();
  broken_points->clear();
  for (int i_sect = 1, n_sect = zone.CountSections(); i_sect <= n_sect; ++i_sect) {
    auto const &section = zone.GetSection(i_sect);
    if (section.type() == CGNS_ENUMV(HEXA_8)) {
      int i_cell_max = section.CellIdMax();
#     pragma omp parallel for num_threads(n_thread)
      for (int i_cell = section.CellIdMin(); i_cell <= i_cell_max; ++i_cell) {
        auto const *nodes = section.GetNodeIdList(i_cell);
        auto coordinate = Coordinate{
            GetGlobal(coordinates, nodes[0]), GetGlobal(coordinates, nodes[1]),
            GetGlobal(coordinates, nodes[2]), GetGlobal(coordinates, nodes[3]),
            GetGlobal(coordinates, nodes[4]), GetGlobal(coordinates, nodes[5]),
            GetGlobal(coordinates, nodes[6]), GetGlobal(coordinates, nodes[7]) };
        auto const &integrator = Integrator(coordinate);
        double det_jac_min = std::numeric_limits<double>::max();
        double det_jac_max = std::numeric_limits<double>::lowest();
        for (int q = 0; q < Integrator::Q; q++) {
          double jacobian = integrator.GetJacobianDeterminant(q);
          det_jac_min = std::min(det_jac_min, jacobian);
          det_jac_max = std::max(det_jac_max, jacobian);
        }
        if (Invalid(det_jac_min, det_jac_max)) {
#         pragma omp critical
          for (int i = 0; i < 8; ++i) {
            broken_points->emplace(nodes[i]);
          }
        }
      }
    }
  }
  auto stop = Clock::now();
  auto cost = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
  std::printf("Finding %ld broken points by %d threads costs %ld seconds\n",
      broken_points->size(), n_thread, cost.count());
  return broken_points->size();
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::printf("usage:\n  %s <old_file> <new_file> <n_thread>\n", argv[0]);
    return -1;
  }
  auto old_file = std::string(argv[1]);
  auto new_file = std::string(argv[2]);
  int n_thread = std::atoi(argv[3]);

  auto old_tree = File(old_file);
  auto new_tree = File(new_file);

  // read the unique CGNSBase_t
  old_tree.ReadBases();
  new_tree.ReadBases();
  if (old_tree.CountBases() != 1 || new_tree.CountBases() != 1) {
    throw std::runtime_error("Only 1-CGNSBase_t CGNSTree_t's are supported!");
  }
  auto &old_base = old_tree.GetBase(1);
  auto &new_base = new_tree.GetBase(1);

  // read the unique Zone_t
  if (old_base.CountZones() != 1 || new_base.CountZones() != 1) {
    throw std::runtime_error("Only 1-Zone_t CGNSBase_t's are supported!");
  }
  auto &old_zone = old_base.GetZone(1);
  auto &new_zone = new_base.GetZone(1);
  auto n_node = old_zone.CountNodes();
  auto n_cell = old_zone.CountCells();
  assert(n_node == new_zone.CountNodes());
  assert(n_cell == new_zone.CountCells());
  std::printf("n_node = %d, n_cell = %d\n", n_node, n_cell);

  // read GridCoodinates_t
  auto &old_coordinates = old_zone.GetCoordinates();
  auto &new_coordinates = new_zone.GetCoordinates();

  // the main loop
  auto broken_points = std::unordered_set<cgsize_t>();
  while (FindBrokenPoints(new_zone, new_coordinates, n_thread, &broken_points)) {
    for (auto i_node : broken_points) {
      // half the shift of this point
      auto shift_x = new_coordinates.x(i_node) - old_coordinates.x(i_node);
      new_coordinates.x(i_node) -= shift_x * 0.5;
      auto shift_y = new_coordinates.y(i_node) - old_coordinates.y(i_node);
      new_coordinates.y(i_node) -= shift_y * 0.5;
      auto shift_z = new_coordinates.z(i_node) - old_coordinates.z(i_node);
      new_coordinates.z(i_node) -= shift_z * 0.5;
    }
  }

  // write the fixed cgns out
  for (int i = 0; i < 5; ++i) {
    new_file.pop_back();
  }
  auto output = new_file + "_fixed.cgns";
  std::printf("writing to %s ...\n", output.c_str());
  new_tree.Write(output);
}