// Copyright 2021 PEI Weicheng and YANG Minghao and JIANG Yuyan

#ifndef MINI_MESH_METIS_HPP_
#define MINI_MESH_METIS_HPP_

#include <concepts>
#include <ranges>

#include <algorithm>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "metis.h"

namespace mini {
namespace mesh {
namespace metis {

/**
 * @brief A CSR (Compressed Sparse Row) representation of sparse graphs.
 * 
 * In this representation, the indices of neighbors of the `i`th vertex are `[ index(range(i)), index(range(i) + 1), ..., index(range(i + 1)) )`, which can be obtained by the `neighbors()` method.
 * 
 * @tparam Int index type
 */
template <std::integral Int>
class SparseGraph {
 protected:
  Int n_vertex_, *range_, *index_;

 public:
  /**
   * @brief Construct a new object.
   * 
   * @param n_vertex the number of vertices
   * @param range the address of the `range` array allocated by someone else
   * @param index the address of the `index` array allocated by someone else
   */
  SparseGraph(Int n_vertex, Int *range, Int *index)
      : n_vertex_(n_vertex), range_(range), index_(index) {
  }
  SparseGraph() = default;

 public:
  Int const &range(Int i) const {
    return range_[i];
  }
  Int const &index(Int i) const {
    return index_[i];
  }
  Int CountVertices() const {
    return n_vertex_;
  }
  Int CountEdges() const {
    return range_[n_vertex_];
  }
  Int &range(Int i) {
    return range_[i];
  }
  Int &index(Int i) {
    return index_[i];
  }
  Int CountNeighbors(Int i) const {
    return range(i + 1) - range(i);
  }
  /**
   * @brief Get a vector-like object that contains `[ index(range(i)), index(range(i) + 1), ..., index(range(i + 1)) )`.
   * 
   * @param i the index of the query vertex
   * @return auto the vector-like object that contains the indices of its neighbors
   */
  std::ranges::input_range auto neighbors(Int i) const {
    auto ptr_view = std::views::iota(index_ + range(i), index_ + range(i + 1));
    return ptr_view | std::views::transform([](auto *ptr){ return *ptr; });
  }
};

/**
 * @brief A wrapper, which holds arrays allocated by METIS, on SparseGraph.
 * 
 * @tparam Int the index type
 */
template <std::integral Int>
class SparseGraphWithDeleter : public SparseGraph<Int> {
 public:
  /**
   * @brief Construct a new SparseGraphWithDeleter object.
   * 
   * @param n_vertex the number of vertices
   * @param range the address of the `range` array allocated by METIS
   * @param index the address of the `index` array allocated by METIS
   */
  SparseGraphWithDeleter(Int n_vertex, Int *range, Int *index)
      : SparseGraph<Int>(n_vertex, range, index) {
  }
  /**
   * @brief Destroy the SparseGraphWithDeleter object
   *
   * Explicitly free the arrays, which are allocated by METIS. 
   */
  ~SparseGraphWithDeleter() noexcept {
    int errors = (METIS_Free(this->range_) != METIS_OK)
               + (METIS_Free(this->index_) != METIS_OK);
    assert(errors == 0);
  }
};

/**
 * @brief A CSR (Compressed Sparse Row) representation of meshes.
 * 
 * In this representation, the global indices of nodes in the `i`th cell are `[ nodes(range(i)), nodes(range(i) + 1), ..., nodes(range(i + 1)) )`.
 * 
 * @tparam Int the index type
 */
template <std::integral Int>
class Mesh {
  std::vector<Int> range_, index_;
  SparseGraph<Int> graph_;
  Int n_node_;

 public:
  Mesh() = default;
  Mesh(Int *range, Int *index, Int n_cell, Int n_node)
      : graph_(n_cell, range, index), n_node_(n_node) {
    assert(CountLocalNodes() == range[n_cell]);
    assert(n_node > *(std::max_element(index, index + CountLocalNodes())));
  }
  Mesh(std::vector<Int> const &range, std::vector<Int> const &index, Int n_node)
      : Mesh(const_cast<Int *>(range.data()), const_cast<Int *>(index.data()),
          range.size() - 1, n_node) {
    assert(CountLocalNodes() == index.size());
  }
  Mesh(std::vector<Int> &&range, std::vector<Int> &&index, Int n_node)
      : range_(std::move(range)), index_(std::move(index)),
        graph_(range_.size() - 1, range_.data(), index_.data()),
        n_node_(n_node) {
    assert(CountLocalNodes() == index_.size());
  }

 public:
  Int const &range(Int i) const {
    return graph_.range(i);
  }
  Int const &nodes(Int i) const {
    return graph_.index(i);
  }
  Int CountCells() const {
    return graph_.CountVertices();
  }
  Int CountLocalNodes() const {
    return graph_.CountEdges();
  }
  Int CountNodes() const {
    return n_node_;
  }
  Int &range(Int i) {
    return graph_.range(i);
  }
  Int &nodes(Int i) {
    return graph_.index(i);
  }
  /**
   * @brief Resize the underling containers.
   * 
   * @param n_cell the number of cells in the resized mesh
   * @param n_node_global the number of nodes counted globally (without duplication)
   * @param n_node_local the number of nodes counted locally (with duplication)
   */
  void resize(Int n_cell, Int n_node_global, Int n_node_local) {
    assert(0 < n_cell);
    assert(0 < n_node_global);
    assert(0 < n_node_local);
    range_.resize(n_cell + 1);
    index_.resize(n_node_local);
    graph_ = SparseGraph<Int>(n_cell, range_.data(), index_.data());
    n_node_ = n_node_global;
  }
  /**
 * @brief A wrapper of `METIS_MeshToDual()`, which converts a Mesh to its dual graph.
   * 
   * @param n_common_nodes the minimum number of nodes shared by two neighboring cells
   * @param index_base the the base of indexing (0 or 1)
   * @return SparseGraphWithDeleter<Int> the dual graph
   */
  auto GetDualGraph(Int n_common_nodes, Int index_base = 0) const {
    Int n_cells = CountCells();
    Int n_nodes = CountNodes();
    Int *vertex_range, *neighbors;
    auto error_code = METIS_MeshToDual(
        &n_cells, &n_nodes,
        const_cast<Int *>(&(range(0))),
        const_cast<Int *>(&(nodes(0))),
        &n_common_nodes, &index_base,
        &vertex_range, &neighbors);
    assert(error_code == METIS_OK);
    return SparseGraphWithDeleter<Int>(n_cells, vertex_range, neighbors);
  }
};

template <class Container>
static inline bool valid(Container &&c, std::size_t size) {
  return c.size() == 0 || c.size() == size;
}

/**
 * @brief A wrapper of `METIS_PartGraphKway()`, which partitions a graph into K parts.
 * 
 * @tparam Int index type
 * 
 * @param[in] graph the graph to be partitioned
 * @param[in] n_parts the number of parts to be partitioned
 * @param[in] n_constraints the number of balancing constraints (>= 1)
 * @param[in] cost_of_each_vertex the computational cost of each vertex
 * @param[in] size_of_each_vertex the communication size of each vertex
 * @param[in] cost_of_each_edge the weight of each edge
 * @param[in] weight_of_each_part the weight of each part (sum must be 1.0)
 * @param[in] unbalances the unbalance tolerance for each constraint
 * @param[in] options the array of METIS options
 * @return the part id of each vertex
 */
template <std::integral Int>
std::vector<Int> PartGraph(
    const SparseGraph<Int> &graph, Int n_parts, Int n_constraints = 1,
    const std::vector<Int> &cost_of_each_vertex = {},
    const std::vector<Int> &size_of_each_vertex = {},
    const std::vector<Int> &cost_of_each_edge = {},
    const std::vector<real_t> &weight_of_each_part = {},
    const std::vector<real_t> &unbalances = {},
    const std::vector<Int> &options = {}) {
  static_assert(sizeof(Int) == sizeof(idx_t),
      "`Int` and `idx_t` must have the same size.");
  Int n_vertices = graph.CountVertices();
  auto vertex_parts = std::vector<Int>(n_vertices);
  if (n_parts == 1)
    return vertex_parts;
  assert(valid(cost_of_each_vertex, n_vertices));
  assert(valid(size_of_each_vertex, n_vertices));
  assert(valid(cost_of_each_edge, graph.CountEdges()));
  assert(valid(weight_of_each_part, n_parts));
  assert(valid(unbalances, n_parts));
  Int objective_value;
  auto error_code = METIS_PartGraphKway(
      &n_vertices, &n_constraints,
      const_cast<Int *>(&(graph.range(0))),
      const_cast<Int *>(&(graph.index(0))),
      const_cast<Int *>(cost_of_each_vertex.data()),
      const_cast<Int *>(size_of_each_vertex.data()),
      const_cast<Int *>(cost_of_each_edge.data()),
      const_cast<Int *>(&n_parts),
      const_cast<real_t *>(weight_of_each_part.data()),
      const_cast<real_t *>(unbalances.data()),
      const_cast<Int *>(options.data()),
      &objective_value, vertex_parts.data());
  assert(error_code == METIS_OK);
  return vertex_parts;
}
/**
 * @brief A wrapper of `METIS_PartMeshDual()`, which partitions a mesh into K parts.
 * 
 * @tparam Int index type
 * @tparam Real real number type
 * 
 * @param[in] mesh the mesh to be partitioned
 * @param[in] n_parts the number of parts to be partitioned
 * @param[in] n_common_nodes the minimum number of nodes shared by two neighboring cells
 * @param[in] cost_of_each_cell the computational cost of each cell
 * @param[in] size_of_each_cell the communication size of each cell
 * @param[in] weight_of_each_part the weight of each part (sum must be 1.0)
 * @param[in] options the array of METIS options
 * @return `{ cell_parts, node_parts }` the part id of each cell and each node
 */
template <std::integral Int>
std::pair<std::vector<Int>, std::vector<Int>> PartMesh(
    const Mesh<Int> &mesh, Int n_parts, Int n_common_nodes = 2,
    const std::vector<Int> &cost_of_each_cell = {},
    const std::vector<Int> &size_of_each_cell = {},
    const std::vector<real_t> &weight_of_each_part = {},
    const std::vector<Int> &options = {}) {
  static_assert(sizeof(Int) == sizeof(idx_t),
      "`Int` and `idx_t` must have the same size.");
  Int n_cells = mesh.CountCells();
  Int n_nodes = mesh.CountNodes();
  assert(valid(cost_of_each_cell, n_cells));
  assert(valid(size_of_each_cell, n_cells));
  assert(valid(weight_of_each_part, n_parts));
  std::vector<Int> cell_parts(n_cells), node_parts(n_nodes);
  Int objective_value;
  auto error_code = METIS_PartMeshDual(
      &n_cells, &n_nodes,
      const_cast<Int *>(&(mesh.range(0))),
      const_cast<Int *>(&(mesh.nodes(0))),
      const_cast<Int *>(cost_of_each_cell.data()),
      const_cast<Int *>(size_of_each_cell.data()),
      &n_common_nodes, &n_parts,
      const_cast<real_t *>(weight_of_each_part.data()),
      const_cast<Int *>(options.data()),
      &objective_value, cell_parts.data(), node_parts.data());
  assert(error_code == METIS_OK);
  return {cell_parts, node_parts};
}

/**
 * @brief Get the partition of nodes from the partition of cells.
 * 
 * @tparam Int 
 * @param mesh 
 * @param cell_parts 
 * @param n_parts 
 * @return std::vector<Int> 
 */
template <std::integral Int>
std::vector<Int> GetNodeParts(
    const metis::Mesh<Int>& mesh, const std::vector<Int>& cell_parts,
    Int n_parts) {
  Int n_nodes = mesh.CountNodes();
  Int n_cells = mesh.CountCells();
  auto node_parts = std::vector<Int>(n_nodes, n_parts);
  auto node_pointer = &(mesh.nodes(0));
  auto curr_range_pointer = &(mesh.range(0));
  for (Int i = 0; i < n_cells; ++i) {
    auto part_value = cell_parts[i];
    auto head = *curr_range_pointer++;
    auto tail = *curr_range_pointer;
    for (Int j = head; j < tail; ++j) {
      Int node_index = *node_pointer++;
      /* If more than one parts share a common node, then the
      node belongs to the min-id part. */
      if (part_value < node_parts[node_index])
        node_parts[node_index] = part_value;
    }
  }
  return node_parts;
}

}  // namespace metis
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_METIS_HPP_
