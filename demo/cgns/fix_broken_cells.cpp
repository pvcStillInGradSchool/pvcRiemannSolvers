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
#include "mini/mesh/quality.hpp"
#include "mini/coordinate/hexahedron.hpp"
#include "mini/integrator/hexahedron.hpp"
#include "mini/integrator/lobatto.hpp"

using Clock = std::chrono::high_resolution_clock;

using File = mini::mesh::cgns::File<double>;
using Zone = mini::mesh::cgns::Zone<double>;
using Coordinates = mini::mesh::cgns::Coordinates<double>;
using Solution = mini::mesh::cgns::Solution<double>;
using Field = mini::mesh::cgns::Field<double>;

using Measure = mini::mesh::QualityMeasure<double>;

inline bool Invalid(double det_jac_min, double det_jac_max) {
  return det_jac_min <= 0 || det_jac_min < det_jac_max * 0.01;
}

bool FindBrokenPoints(Zone const &zone, Coordinates const &coordinates, int n_thread,
    std::unordered_set<cgsize_t> *broken_points) {
  auto start = Clock::now();
  broken_points->clear();
  for (int i_sect = 1, n_sect = zone.CountSections(); i_sect <= n_sect; ++i_sect) {
    auto const &section = zone.GetSection(i_sect);
    auto *Build = Measure::SelectBuilder(section.type());
    if (Build) {
      int i_cell_max = section.CellIdMax();
      int npe = mini::mesh::cgns::CountNodesByType(section.type());
#     pragma omp parallel for num_threads(n_thread)
      for (int i_cell = section.CellIdMin(); i_cell <= i_cell_max; ++i_cell) {
        auto const *nodes = section.GetNodeIdList(i_cell);
        auto const &integrator = Build(coordinates, nodes);
        double det_jac_min = std::numeric_limits<double>::max();
        double det_jac_max = std::numeric_limits<double>::lowest();
        for (int q = 0; q < Measure::Integrator::Q; q++) {
          double jacobian = integrator.GetJacobianDeterminant(q);
          det_jac_min = std::min(det_jac_min, jacobian);
          det_jac_max = std::max(det_jac_max, jacobian);
        }
        if (Invalid(det_jac_min, det_jac_max)) {
#         pragma omp critical
          for (int i = 0; i < npe; ++i) {
            broken_points->emplace(nodes[i]);
          }
        }
      }
    }
  }
  auto stop = Clock::now();
  auto cost = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::printf("Finding %ld broken points by %d threads costs %ld milliseconds\n",
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
  auto &old_base = old_tree.GetUniqueBase();
  auto &new_base = new_tree.GetUniqueBase();

  // read the unique Zone_t
  auto &old_zone = old_base.GetUniqueZone();
  auto &new_zone = new_base.GetUniqueZone();
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

  // update the vector field of point shift
  Field *shift_x{nullptr}, *shift_y{nullptr}, *shift_z{nullptr};
  for (int i_sol = 1, n_sol = new_zone.CountSolutions(); i_sol <= n_sol; i_sol++) {
    auto &solution = new_zone.GetSolution(i_sol);
    if (solution.localtion() == CGNS_ENUMV(Vertex)) {
      for (int i_field = 1, n_field = solution.CountFields(); i_field <= n_field; ++i_field) {
        auto &field = solution.GetField(i_field);
        if (field.name() == "ShiftX") {
          shift_x = &field;
          continue;
        }
        if (field.name() == "ShiftY") {
          shift_y = &field;
          continue;
        }
        if (field.name() == "ShiftZ") {
          shift_z = &field;
          continue;
        }
      }
    }
  }
  assert(shift_x && shift_y && shift_z);
  for (cgsize_t i_node = 1; i_node <= n_node; ++i_node) {
    shift_x->at(i_node) = new_coordinates.x(i_node) - old_coordinates.x(i_node);
    shift_y->at(i_node) = new_coordinates.y(i_node) - old_coordinates.y(i_node);
    shift_z->at(i_node) = new_coordinates.z(i_node) - old_coordinates.z(i_node);
  }

  // write the fixed cgns out
  for (int i = 0; i < 5; ++i) {
    new_file.pop_back();
  }
  auto output = new_file + "_fixed.cgns";
  std::printf("writing to %s ...\n", output.c_str());
  new_tree.Write(output);
}
