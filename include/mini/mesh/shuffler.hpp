// Copyright 2020 PEI Weicheng and YANG Minghao

#ifndef MINI_MESH_SHUFFLER_HPP_
#define MINI_MESH_SHUFFLER_HPP_

#include <concepts>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mini/mesh/cgns.hpp"
#include "mini/mesh/metis.hpp"
#include "mini/mesh/mapper.hpp"

namespace mini {
namespace mesh {

/**
 * @brief Get the New Order object
 * 
 * @tparam T 
 * @param parts 
 * @param n 
 * @return a pair of std::vector<int>
 */
template <typename T>
std::pair<std::vector<int>, std::vector<int>>
GetNewOrder(T const *parts, int n) {
  auto new_to_old = std::vector<int>(n);
  auto old_to_new = std::vector<int>(n);
    // TODO(PVC): use std::ranges::view in C++23
  std::iota(new_to_old.begin(), new_to_old.end(), 0);
  auto cmp = [parts](int lid, int rid) {
    return parts[lid] < parts[rid] || (parts[lid] == parts[rid] && lid < rid);
  };
  std::sort(new_to_old.begin(), new_to_old.end(), cmp);
  for (int i = 0; i < n; ++i)
    old_to_new[new_to_old[i]] = i;
  return {new_to_old, old_to_new};
}
/**
 * @brief 
 * 
 * @tparam T 
 * @param new_to_old 
 * @param old_data 
 */
template <typename T>
void ShuffleData(std::vector<int> const &new_to_old, T *old_data) {
  int n = new_to_old.size();
  std::vector<T> new_data(n);
  for (int i = 0; i < n; ++i)
    new_data[i] = old_data[new_to_old[i]];
  std::memcpy(old_data, new_data.data(), n * sizeof(T));
}
// template <typename T>
// void UpdateNodeIdList(std::vector<int> const &old_to_new_for_nodes,
//                       int list_length, T *node_id_list) {
//   auto *tail = node_id_list + list_length;
//   auto *curr = node_id_list;
//   while (curr != tail) {
//     auto i_node_old = *curr;
//     auto i_node_new = old_to_new_for_nodes[i_node_old];
//     *curr++ = i_node_new;
//   }
// }
/**
 * @brief 
 * 
 * @tparam T 
 * @param old_to_new_for_nodes 
 * @param new_to_old_for_cells 
 * @param npe 
 * @param old_cid_old_nid 
 */
template <typename T>
void ShuffleConnectivity(std::vector<int> const &old_to_new_for_nodes,
                         std::vector<int> const &new_to_old_for_cells,
                         int npe, T *old_cid_old_nid) {
  int n_cells = new_to_old_for_cells.size();
  int i_node_list_size = n_cells * npe;
  auto old_cid_new_nid = std::vector<T>(i_node_list_size);
  for (int i = 0; i < i_node_list_size; ++i) {
    auto old_nid = old_cid_old_nid[i];
    auto new_nid = old_to_new_for_nodes.at(old_nid - 1) + 1;
    old_cid_new_nid[i] = new_nid;
  }
  auto *new_cid_new_nid = old_cid_old_nid;
  for (int new_cid = 0; new_cid < n_cells; ++new_cid) {
    int range_min = npe * new_to_old_for_cells[new_cid];
    auto old_ptr = old_cid_new_nid.data() + range_min;
    for (int i = 0; i < npe; ++i) {
      *new_cid_new_nid++ = *old_ptr++;
    }
  }
}

/**
 * @brief 
 * 
 * @tparam Int
 * @tparam Real 
 */
template <std::integral Int, std::floating_point Real>
class Shuffler {
 public:
  using CgnsMesh = mini::mesh::cgns::File<Real>;
  using MetisMesh = metis::Mesh<Int>;
  using Graph = metis::SparseGraphWithDeleter<Int>;
  using Mapper = mini::mesh::mapper::CgnsToMetis<Int, Real>;
  using Section = mini::mesh::cgns::Section<Real>;
  using Solution = mini::mesh::cgns::Solution<Real>;
  using Field = mini::mesh::cgns::Field<Real>;

  Shuffler(Int n_parts, std::vector<Int> const &cell_parts,
           std::vector<Int> const &node_parts,
           Graph const &graph, MetisMesh const &metis_mesh,
           CgnsMesh *cgns_mesh, Mapper *mapper)
      : n_parts_{n_parts}, cell_parts_{cell_parts}, node_parts_{node_parts},
        graph_(graph), metis_mesh_(metis_mesh), cgns_mesh_(cgns_mesh),
        mapper_(mapper) {
  }

  void Shuffle();
  void WritePartitionInfo(std::string const &case_name);
  static void PartitionAndShuffle(std::string const &case_name,
      std::string const &old_cgns_name, Int n_parts);

 private:
  std::vector<Int> const &cell_parts_;
  std::vector<Int> const &node_parts_;
  std::map<Int, std::map<Int, cgns::ShiftedVector<Int>>> face_parts_;
  CgnsMesh *cgns_mesh_;
  Mapper *mapper_;
  Graph const &graph_;
  MetisMesh const &metis_mesh_;
  Int n_parts_;

  void FillFaceParts(CgnsMesh const &mesh, Mapper const &mapper);
};

template <std::integral Int, std::floating_point Real>
void Shuffler<Int, Real>::FillFaceParts(
    CgnsMesh const &mesh, Mapper const &mapper) {
  auto &m_to_c_nodes = mapper.metis_to_cgns_for_nodes;
  auto &m_to_c_cells = mapper.metis_to_cgns_for_cells;
  auto &c_to_m_nodes = mapper.cgns_to_metis_for_nodes;
  auto &c_to_m_cells = mapper.cgns_to_metis_for_cells;
  auto &base = mesh.GetBase(1);
  auto n_zones = base.CountZones();
  // For each node, find its user cells and append them to its vector:
  auto node_user_cells = std::vector<std::vector<Int>>(node_parts_.size());
  for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
    auto &zone = base.GetZone(i_zone);
    auto &i_node_to_m_node = c_to_m_nodes.at(i_zone);
    auto n_sects = zone.CountSections();
    for (int i_sect = 1; i_sect <= n_sects; ++i_sect) {
      auto &sect = zone.GetSection(i_sect);
      if (sect.dim() == base.GetCellDim()) {
        auto npe = sect.CountNodesByType();
        auto n_cells = sect.CountCells();
        auto i_cell_min = sect.CellIdMin();
        auto i_cell_max = sect.CellIdMax();
        auto &i_cell_to_m_cell = c_to_m_cells.at(i_zone).at(i_sect);
        auto *head = sect.GetNodeIdList();
        auto *curr = head;
        auto *tail = head + n_cells * npe;
        for (int i_cell = i_cell_min; i_cell <= i_cell_max; ++i_cell) {
          auto m_cell = i_cell_to_m_cell.at(i_cell);
          for (int k = 0; k < npe; ++k) {
            assert(curr + k < tail);
            auto m_node = i_node_to_m_node.at(curr[k]);
            node_user_cells[m_node].emplace_back(m_cell);
          }
          curr += npe;
        }
        assert(curr == tail);
      }
    }
  }
  // For each face, determine its part by its user cells:
  for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
    auto &zone = base.GetZone(i_zone);
    auto &i_node_to_m_node = c_to_m_nodes.at(i_zone);
    auto n_sects = zone.CountSections();
    for (int i_sect = 1; i_sect <= n_sects; ++i_sect) {
      auto &sect = zone.GetSection(i_sect);
      if (sect.dim() + 1 == base.GetCellDim()) {
        auto npe = sect.CountNodesByType();
        auto n_faces = sect.CountCells();
        auto i_face_min = sect.CellIdMin();
        auto i_face_max = sect.CellIdMax();
        auto &i_face_to_part = (face_parts_[i_zone][i_sect]
            = cgns::ShiftedVector<Int>(n_faces, i_face_min));
        auto *head = sect.GetNodeIdList();
        auto *curr = head;
        auto *tail = head + n_faces * npe;
        // For each face, count the user cells:
        for (int i_face = i_face_min; i_face <= i_face_max; ++i_face) {
          auto face_user_cell_cnts = std::unordered_map<Int, Int>();
          for (int k = 0; k < npe; ++k) {
            assert(curr + k < tail);
            auto i_node = curr[k];
            auto m_node = i_node_to_m_node.at(i_node);
            for (Int m_cell : node_user_cells[m_node]) {
              face_user_cell_cnts[m_cell]++;
            }
          }
          std::vector<int> npe_cells;
          // Find the cell that uses this face npe times:
          for (auto [m_cell, cnt] : face_user_cell_cnts) {
            assert(cnt <= npe);
            if (cnt == npe) {
              i_face_to_part.at(i_face) = cell_parts_.at(m_cell);
              npe_cells.emplace_back(m_cell);
            }
          }
          // There should be one and only one such user:
          assert(npe_cells.size() == 1);
          if (npe_cells.size() > 1) {
            for (auto m_cell : npe_cells)
              std::printf("face[%d] is owned by cell[%d] for %d times.\n",
                  i_face, static_cast<int>(m_cell), npe);
          }
          curr += npe;
        }
        assert(curr == tail);
      }
    }
  }
}

template <std::integral Int, std::floating_point Real>
void Shuffler<Int, Real>::Shuffle() {
  FillFaceParts(*cgns_mesh_, *mapper_);
  auto &m_to_c_nodes = mapper_->metis_to_cgns_for_nodes;
  auto &m_to_c_cells = mapper_->metis_to_cgns_for_cells;
  auto &c_to_m_nodes = mapper_->cgns_to_metis_for_nodes;
  auto &c_to_m_cells = mapper_->cgns_to_metis_for_cells;
  auto &base = cgns_mesh_->GetBase(1);
  auto n_zones = base.CountZones();
  // shuffle nodes, cells and data on them
  for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
    auto &zone = base.GetZone(i_zone);
    // shuffle node coordinates
    auto metis_nid_offset = c_to_m_nodes[i_zone][1];
    auto [new_to_old_for_nodes, old_to_new_for_nodes] = GetNewOrder(
        &(node_parts_[metis_nid_offset]), zone.CountNodes());
    auto &coord = zone.GetCoordinates();
    ShuffleData(new_to_old_for_nodes, coord.x().data());
    ShuffleData(new_to_old_for_nodes, coord.y().data());
    ShuffleData(new_to_old_for_nodes, coord.z().data());
    ShuffleData(new_to_old_for_nodes, &(c_to_m_nodes[i_zone][1]));
    ShuffleData(old_to_new_for_nodes, &(m_to_c_nodes[metis_nid_offset]));
    // shuffle data on nodes
    auto n_solns = zone.CountSolutions();
    for (auto i_soln = 1; i_soln <= n_solns; i_soln++) {
      auto &solution = zone.GetSolution(i_soln);
      if (!solution.OnNodes())
        continue;
      for (auto i = 1; i <= solution.CountFields(); ++i) {
        auto &field = solution.GetField(i);
        ShuffleData(new_to_old_for_nodes, field.data());
      }
    }
    // shuffle cells and data on cells
    auto n_sects = zone.CountSections();
    for (auto i_sect = 1; i_sect <= n_sects; ++i_sect) {
      auto &section = zone.GetSection(i_sect);
      /* Shuffle Connectivity */
      auto npe = section.CountNodesByType();
      auto n_cells = section.CountCells();
      auto *i_node_list = section.GetNodeIdList();
      if (section.dim() != base.GetCellDim()) {
        // For a lower-dim section, get each face's partition
        auto [new_to_old_for_faces, old_to_new_for_faces] = GetNewOrder(
            face_parts_.at(i_zone).at(i_sect).data(), section.CountCells());
        // then shuffle its connectivity
        ShuffleConnectivity(old_to_new_for_nodes, new_to_old_for_faces,
            npe, i_node_list);
        continue;
      }
      auto range_min = section.CellIdMin();
      auto metis_cid_offset = c_to_m_cells[i_zone][i_sect].at(range_min);
      auto [new_to_old_for_cells, old_to_new_for_cells] = GetNewOrder(
          &(cell_parts_[metis_cid_offset]), n_cells);
      ShuffleData(old_to_new_for_cells, &(m_to_c_cells[metis_cid_offset]));
      ShuffleData(new_to_old_for_cells, c_to_m_cells[i_zone][i_sect].data());
      ShuffleConnectivity(old_to_new_for_nodes, new_to_old_for_cells,
          npe, i_node_list);
      /* Shuffle Data on Cells */
      auto n_solns = zone.CountSolutions();
      for (auto i_soln = 1; i_soln <= n_solns; i_soln++) {
        auto &solution = zone.GetSolution(i_soln);
        if (!solution.OnCells())
          continue;
        for (auto i = 1; i <= solution.CountFields(); ++i) {
          auto &field = solution.GetField(i);
          auto *field_ptr = &(field.at(range_min));
          ShuffleData(new_to_old_for_cells, field_ptr);
        }
      }
    }
    zone.UpdateSectionRanges();
  }
}

template <std::integral Int, std::floating_point Real>
void Shuffler<Int, Real>::WritePartitionInfo(std::string const &case_name) {
  auto &base = cgns_mesh_->GetBase(1);
  int n_zones = base.CountZones();
  /* Prepare to-be-written info for each part: */
  // auto [i_node_min, i_node_max] = part_to_nodes[i_part][i_zone];
  // auto [i_cell_min, i_cell_max] = part_to_cells[i_part][i_zone][i_sect];
  // auto [i_face_min, i_face_max] = part_to_faces[i_part][i_zone][i_sect];
  auto part_to_nodes = std::vector<std::vector<std::pair<int, int>>>(n_parts_);
  auto part_to_cells
      = std::vector<std::vector<std::vector<std::pair<int, int>>>>(n_parts_);
  auto part_to_faces
      = std::vector<std::vector<std::vector<std::pair<int, int>>>>(n_parts_);
  for (int p = 0; p < n_parts_; ++p) {
    part_to_nodes[p].resize(n_zones + 1);
    part_to_cells[p].resize(n_zones + 1);
    part_to_faces[p].resize(n_zones + 1);
    for (int z = 1; z <= n_zones; ++z) {
      part_to_cells[p][z].resize(base.GetZone(z).CountSections() + 1);
      part_to_faces[p][z].resize(base.GetZone(z).CountSections() + 1);
    }
  }
  for (auto &[i_zone, zone] : face_parts_) {
    for (auto &[i_sect, parts] : zone) {
      std::sort(parts.begin(), parts.end());
    }
  }
  /* Get index range of each zone's nodes, cells and faces: */
  for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
    auto &zone = base.GetZone(i_zone);
    auto &node_field = zone.GetSolution("DataOnNodes").GetField("PartIndex");
    // slice node lists by i_part
    int prev_nid = 1, prev_part = node_field.at(prev_nid);
    int n_nodes = zone.CountNodes();
    for (int curr_nid = prev_nid + 1; curr_nid <= n_nodes; ++curr_nid) {
      int curr_part = node_field.at(curr_nid);
      if (curr_part != prev_part) {
        part_to_nodes[prev_part][i_zone] = { prev_nid, curr_nid };
        prev_nid = curr_nid;
        prev_part = curr_part;
      }
    }  // for each node
    part_to_nodes[prev_part][i_zone] = { prev_nid, n_nodes + 1 };
    // slice cell lists by i_part
    int n_cells = zone.CountCells();
    auto &cell_field = zone.GetSolution("DataOnCells").GetField("PartIndex");
    for (int i_sect = 1; i_sect <= zone.CountSections(); ++i_sect) {
      auto &sect = zone.GetSection(i_sect);
      if (sect.dim() != base.GetCellDim())
        continue;
      int cid_min = sect.CellIdMin(), cid_max = sect.CellIdMax();
      int prev_cid = cid_min, prev_part = cell_field.at(prev_cid);
      for (int curr_cid = prev_cid + 1; curr_cid <= cid_max; ++curr_cid) {
        int curr_part = cell_field.at(curr_cid);
        if (curr_part != prev_part) {  // find a new part
          part_to_cells[prev_part][i_zone][i_sect] = { prev_cid, curr_cid };
          prev_cid = curr_cid;
          prev_part = curr_part;
        }
      }  // for each sell
      part_to_cells[prev_part][i_zone][i_sect] = { prev_cid, cid_max + 1 };
    }  // for each sect of cells
    // slice face lists by i_part
    for (int i_sect = 1; i_sect <= zone.CountSections(); ++i_sect) {
      auto &sect = zone.GetSection(i_sect);
      auto &face_field = face_parts_[i_zone][i_sect];
      if (sect.dim() + 1 != base.GetCellDim())
        continue;
      int fid_min = sect.CellIdMin(), fid_max = sect.CellIdMax();
      int prev_fid = fid_min, prev_part = face_field.at(prev_fid);
      for (int curr_fid = prev_fid + 1; curr_fid <= fid_max; ++curr_fid) {
        int curr_part = face_field.at(curr_fid);
        if (curr_part != prev_part) {  // find a new part
          part_to_faces[prev_part][i_zone][i_sect] = { prev_fid, curr_fid };
          prev_fid = curr_fid;
          prev_part = curr_part;
        }
      }  // for each face
      part_to_faces[prev_part][i_zone][i_sect] = { prev_fid, fid_max + 1 };
    }  // for each sect of faces
  }  // for each zone
  /* Store cell adjacency for each part: */
  // inner_adjs[i_part] = std::vector of [i_cell_small, i_cell_large]
  auto inner_adjs = std::vector<std::vector<std::pair<int, int>>>(n_parts_);
  auto part_interpart_adjs
      = std::vector<std::map<int, std::vector<std::pair<int, int>>>>(n_parts_);
  auto part_adj_nodes = std::vector<std::map<int, std::set<int>>>(n_parts_);
  auto sendp_recvp_nodes = std::vector<std::map<int, std::set<int>>>(n_parts_);
  for (int i = 0; i < metis_mesh_.CountCells(); ++i) {
    auto part_i = cell_parts_[i];
    int range_b = metis_mesh_.range(i), range_e = metis_mesh_.range(i + 1);
    for (int i_range = range_b; i_range < range_e; ++i_range) {
      auto i_node = metis_mesh_.nodes(i_range);
      auto node_part = node_parts_[i_node];
      if (node_part != part_i) {
        part_adj_nodes[part_i][node_part].emplace(i_node);
        sendp_recvp_nodes[node_part][part_i].emplace(i_node);
      }
    }
    for (int r = graph_.range(i); r < graph_.range(i + 1); ++r) {
      int j = graph_.index(r);
      auto part_j = cell_parts_[j];
      if (part_i == part_j) {
        if (i < j)
          inner_adjs[part_i].emplace_back(i, j);
      } else {
        part_interpart_adjs[part_i][part_j].emplace_back(i, j);
        int range_b = metis_mesh_.range(j), range_e = metis_mesh_.range(j + 1);
        for (int i_range = range_b; i_range < range_e; ++i_range) {
          auto i_node = metis_mesh_.nodes(i_range);
          auto node_part = node_parts_[i_node];
          if (node_part != part_i) {
            part_adj_nodes[part_i][node_part].emplace(i_node);
            sendp_recvp_nodes[node_part][part_i].emplace(i_node);
          }
        }
      }
    }
  }
  /* Write part info to txts: */
  for (int p = 0; p < n_parts_; ++p) {
    auto ostrm = std::ofstream(case_name + "/partition/" + std::to_string(p)
        + ".txt"/*, std::ios::binary */);
    // node ranges
    ostrm << "# i_zone i_node_head i_node_tail\n";
    for (int z = 1; z <= n_zones; ++z) {
      auto [head, tail] = part_to_nodes[p][z];
      if (true) {
        ostrm << z << ' ' << head << ' ' << tail << '\n';
      }
    }
    // send nodes info
    ostrm << "# i_part i_node_metis\n";
    for (auto &[recv_pid, nodes] : sendp_recvp_nodes[p]) {
      for (auto i : nodes) {
        ostrm << recv_pid << ' ' << i << '\n';
      }
    }
    // adjacent nodes
    ostrm << "# i_part i_node_metis i_zone i_node\n";
    for (auto &[i_part, nodes] : part_adj_nodes[p]) {
      for (auto mid : nodes) {
        auto &info = mapper_->metis_to_cgns_for_nodes[mid];
        int zid = info.i_zone, nid = info.i_node;
        ostrm << i_part << ' ' << mid << ' ' << zid << ' ' << nid << '\n';
      }
    }
    // cell ranges
    ostrm << "# i_zone i_sect i_cell_head i_cell_tail\n";
    for (int z = 1; z <= n_zones; ++z) {
      auto n_sects = part_to_cells[p][z].size() - 1;
      for (int s = 1; s <= n_sects; ++s) {
        auto [head, tail] = part_to_cells[p][z][s];
        if (base.GetZone(z).GetSection(s).dim() == base.GetCellDim()) {
          ostrm << z << ' ' << s << ' ' << head << ' ' << tail << '\n';
        }
      }
    }
    // inner adjacency
    ostrm << "# i_cell_metis j_cell_metis\n";
    for (auto [i, j] : inner_adjs[p]) {
      ostrm << i << ' ' << j << '\n';
    }
    // interpart adjacency
    ostrm << "# i_part i_cell_metis j_cell_metis i_node_npe j_node_npe\n";
    for (auto &[i_part, pairs] : part_interpart_adjs[p]) {
      for (auto [i, j] : pairs) {
        auto &info_i = mapper_->metis_to_cgns_for_cells[i];
        auto &info_j = mapper_->metis_to_cgns_for_cells[j];
        int npe_i = base.GetZone(info_i.i_zone).GetSection(info_i.i_sect).
            CountNodesByType();
        int npe_j = base.GetZone(info_j.i_zone).GetSection(info_j.i_sect).
            CountNodesByType();
        ostrm << i_part << ' ' << i << ' ' << j << ' ' << npe_i << ' ' <<
            npe_j << '\n';
      }
    }
    // face ranges
    ostrm << "# i_zone i_sect i_face_head i_face_tail\n";
    for (int z = 1; z <= n_zones; ++z) {
      auto n_sects = part_to_faces[p][z].size() - 1;
      for (int s = 1; s <= n_sects; ++s) {
        auto [head, tail] = part_to_faces[p][z][s];
        if (base.GetZone(z).GetSection(s).dim() + 1 == base.GetCellDim()) {
          ostrm << z << ' ' << s << ' ' << head << ' ' << tail << '\n';
        }
      }
    }
    ostrm << "#\n";
  }
}

template <std::integral Int, std::floating_point Real>
void Shuffler<Int, Real>::PartitionAndShuffle(std::string const &case_name,
    std::string const &old_cgns_name, Int n_parts) {
  char cmd[1024];
  std::snprintf(cmd, sizeof(cmd), "mkdir -p %s/partition",
      case_name.c_str());
  if (std::system(cmd))
    throw std::runtime_error(cmd + std::string(" failed."));
  std::printf("[Done] %s\n", cmd);
  auto cgns_mesh = cgns::File<Real>(old_cgns_name);
  cgns_mesh.ReadBases();
  std::printf("[Done] %s\n", "cgns::File::ReadBases");
  /* Partition the mesh: */
  auto mapper = mapper::CgnsToMetis<Int, Real>();
  metis::Mesh<Int> metis_mesh = mapper.Map(cgns_mesh);
  assert(mapper.IsValid());
  std::printf("[Done] %s\n", "mapper::CgnsToMetis");
  Int n_common_nodes{3};
  auto graph = metis_mesh.GetDualGraph(n_common_nodes);
  std::printf("[Done] %s\n", "metis::Mesh::GetDualGraph");
  auto cell_parts = metis::PartGraph(graph, n_parts);
  std::printf("[Done] %s\n", "metis::PartGraph");
  auto node_parts = metis::GetNodeParts(metis_mesh, cell_parts, n_parts);
  std::printf("[Done] %s\n", "metis::GetNodeParts");
  mapper.WriteParts(cell_parts, node_parts, &cgns_mesh);
  std::printf("[Done] partition `%s` into %d parts.\n",
      old_cgns_name.c_str(), static_cast<int>(n_parts));
  /* Shuffle nodes and cells: */
  auto shuffler = Shuffler<idx_t, double>(n_parts, cell_parts, node_parts,
      graph, metis_mesh, &cgns_mesh, &mapper);
  shuffler.Shuffle();
  assert(mapper.IsValid());
  auto new_cgns_name = case_name + "/shuffled.cgns";
  cgns_mesh.Write(new_cgns_name, 2);
  shuffler.WritePartitionInfo(case_name);
  std::printf("[Done] the %d-part `./%s` has been shuffled.\n",
      static_cast<int>(n_parts), new_cgns_name.c_str());
}

}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_SHUFFLER_HPP_
