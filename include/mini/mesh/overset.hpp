// Copyright 2023 PEI Weicheng
#ifndef MINI_MESH_OVERSET_HPP_
#define MINI_MESH_OVERSET_HPP_

#include <concepts>

#include <algorithm>
#include <cassert>
#include <vector>
#include <utility>
#include <tuple>
#include <unordered_set>

#include "mini/mesh/cgal.hpp"
#include "mini/mesh/cgns.hpp"
#include "mini/mesh/metis.hpp"
#include "mini/mesh/mapper.hpp"


namespace mini {
namespace mesh {
namespace overset {

enum class Status {
  kUnknown, kFringe, kDonor, kHole,
};

template <std::integral Int, std::floating_point Real>
class Mapping {
 public:
  using Mesh = cgns::File<Real>;
  using Graph = metis::SparseGraph<Int>;
  using Mapper = mapper::CgnsToMetis<Int, Real>;
  using Tree = cgal::NeighborSearching<Real>;

  template <std::ranges::input_range R>
  static void AddCellStatus(Status status, R &&cells,
      Mesh *mesh, Graph const &graph, Mapper const &mapper) {
    auto &zone = mesh->GetBase(1).GetZone(1);
    auto &solution = zone.AddSolution("CellData", CGNS_ENUMV(CellCenter));
    auto &field = solution.AddField("OversetStatus");
    for (auto i_cell : cells) {
      auto &index = mapper.metis_to_cgns_for_cells[i_cell];
      field.at(index.i_cell) = static_cast<Real>(status);
    }
  }

  /**
   * @brief Fine fringe cells in the foreground mesh.
   * 
   * @param n_layer how many layers of cells are treated as fringe
   * @param mesh the foreground mesh
   * @param graph the dual graph of the foreground mesh
   * @param mapper the mapper between the two representations
   * @return std::vector<Int> the metis indices of all fringe cells
   */
  static std::vector<Int> FindForegroundFringeCells(int n_layer,
      Mesh const &mesh, Graph const &graph, Mapper const &mapper) {
    std::vector<Int> result;
    assert(mesh.CountBases() == 1);
    auto &base = mesh.GetBase(1);
    // Find the outmost layer according to #neighbors < #faces:
    for (Int i = 0, n = graph.CountVertices(); i < n; ++i) {
      auto cell = mapper.metis_to_cgns_for_cells[i];
      auto &sect = base.GetZone(cell.i_zone).GetSection(cell.i_sect);
      if (graph.CountNeighbors(i) < sect.CountFacesByType()) {
        result.push_back(i);
      }
    }
    // Run breadth-first search for more layers:
    auto curr_begin = result.begin(), curr_end = result.end();
    auto found = std::unordered_set<Int>(curr_begin, curr_end);
    while (curr_begin != curr_end && --n_layer > 0) {
      auto next = std::vector<Int>();
      while (curr_begin != curr_end) {
        auto i = *curr_begin++;  // for each i in current layer
        for (auto j : graph.neighbors(i)) {
          auto iter = found.find(j);
          if (iter == found.end()) {
            found.emplace_hint(iter, j);
            next.push_back(j);
          }
        }
      }
      result.resize(result.size() + next.size());
      curr_end = result.end();
      curr_begin = curr_end - next.size();
      std::ranges::copy(next, curr_begin);
    }
    return result;
  }

  /**
   * @brief Build a spatial search tree for cells.
   * 
   * The tree uses cell centers as the keys for searching.
   * 
   * @param mesh the mesh to be searched
   * @param graph the dual graph of the foreground mesh
   * @param mapper the mapper between the two representations
   * @return Tree the object that supports fast spatial search
   */
  static Tree BuildCellSearchTree(
      Mesh const &mesh, Graph const &graph, Mapper const &mapper) {
    auto n_cell = graph.CountVertices();
    std::vector<Real> x(n_cell), y(n_cell), z(n_cell);
    assert(mesh.CountBases() == 1);
    auto &base = mesh.GetBase(1);
    for (Int i_cell = 0; i_cell < n_cell; ++i_cell) {
      auto &index = mapper.metis_to_cgns_for_cells[i_cell];
      auto &sect = base.GetZone(index.i_zone).GetSection(index.i_sect);
      sect.GetCellCenter(index.i_cell, &x[i_cell], &y[i_cell], &z[i_cell]);
    }
    return Tree(x, y, z);
  }

  /**
   * @brief Find a given number of donor cells in background for each fringe cell in foreground.
   * 
   * @param mesh_fg the foreground mesh
   * @param graph_fg the dual graph of the foreground mesh
   * @param mapper_fg the mapper between the two representations
   * @param fringe_fg the indices of fringe cells in the foreground mesh
   * @param tree_bg the spatial search tree of the background mesh
   * @param n_donor the number of donors of each fringe cell
   * @return std::vector<std::vector<int>> 
   */
  static std::vector<std::vector<int>> FindBackgroundDonorCells(
      Mesh const &mesh_fg, Graph const &graph_fg, Mapper const &mapper_fg,
      std::vector<Int> const &fringe_fg, Tree const &tree_bg, int n_donor) {
    auto result = std::vector<std::vector<int>>();
    result.reserve(fringe_fg.size());
    auto &base = mesh_fg.GetBase(1);
    for (auto i_cell : fringe_fg) {
      auto &index = mapper_fg.metis_to_cgns_for_cells[i_cell];
      auto &sect = base.GetZone(index.i_zone).GetSection(index.i_sect);
      Real x, y, z;
      sect.GetCellCenter(index.i_cell, &x, &y, &z);
      result.emplace_back(tree_bg.Search(x, y, z, n_donor));
    }
    assert(result.size() == fringe_fg.size());
    return result;
  }

  /**
   * @brief Find all donor cells within a sphere in background for each fringe cell in foreground.
   * 
   * @param mesh_fg the foreground mesh
   * @param graph_fg the dual graph of the foreground mesh
   * @param mapper_fg the mapper between the two representations of the foreground mesh
   * @param fringe_fg the indices of fringe cells in the foreground mesh
   * @param tree_bg the spatial search tree of the background mesh
   * @param mesh_bg the background mesh
   * @param graph_bg the dual graph of the background mesh
   * @param mapper_bg the mapper between the two representations of the background mesh
   * @param radius the radius of the bounding ball of each fringe cell
   * @return std::vector<std::vector<int>> 
   */
  static std::vector<std::vector<int>> FindBackgroundDonorCells(
      Mesh const &mesh_fg, Graph const &graph_fg, Mapper const &mapper_fg,
      std::vector<Int> const &fringe_fg, Tree const &tree_bg,
      Mesh const &mesh_bg, Graph const &graph_bg, Mapper const &mapper_bg,
      Real radius) {
    auto result = std::vector<std::vector<int>>();
    result.reserve(fringe_fg.size());
    auto &base = mesh_fg.GetBase(1);
    for (auto i_cell : fringe_fg) {
      auto &index = mapper_fg.metis_to_cgns_for_cells[i_cell];
      auto &sect = base.GetZone(index.i_zone).GetSection(index.i_sect);
      Real x, y, z;
      sect.GetCellCenter(index.i_cell, &x, &y, &z);
      result.emplace_back(tree_bg.Search(x, y, z, 1));
      // Run breadth-first search for more donor cells:
      auto &curr = result.back();
      auto curr_begin = curr.begin(), curr_end = curr.end();
      auto found = std::unordered_set<Int>(curr_begin, curr_end);
      while (curr_begin != curr_end) {
        auto next = std::vector<Int>();
        while (curr_begin != curr_end) {
          auto i = *curr_begin++;  // for each i in current layer
          for (auto j : graph_bg.neighbors(i)) {
            auto iter = found.find(j);
            if (iter != found.end()) {
              continue;  // skip cells that are already found
            }
            auto &index_j = mapper_bg.metis_to_cgns_for_cells[j];
            auto &zone_j = mesh_bg.GetBase(1).GetZone(index_j.i_zone);
            auto &sect_j = zone_j.GetSection(index_j.i_sect);
            Real x_j, y_j, z_j;
            sect_j.GetCellCenter(index_j.i_cell, &x_j, &y_j, &z_j);
            if (radius < std::hypot(x - x_j, y - y_j, z - z_j)) {
              continue;  // skip cells that are out of the bounding ball
            }
            found.emplace_hint(iter, j);
            next.push_back(j);
          }
        }
        curr.resize(curr.size() + next.size());
        curr_end = curr.end();
        curr_begin = curr_end - next.size();
        std::ranges::copy(next, curr_begin);
      }
    }
    assert(result.size() == fringe_fg.size());
    return result;
  }

  /**
   * @brief Merge separated lists of indices into a single set.
   * 
   * @param indices the lists to be merged
   * @return std::unordered_set<Int> the merged set of indices
   */
  static std::unordered_set<Int> merge(
      std::vector<std::vector<int>> const &indices) {
    auto result = std::unordered_set<Int>();
    for (auto &row : indices) {
      for (auto i : row) {
        result.insert(i);
      }
    }
    return result;
  }
};

}  // namespace overset
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_OVERSET_HPP_
