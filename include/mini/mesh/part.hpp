// Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_MESH_PART_HPP_
#define MINI_MESH_PART_HPP_

#include <concepts>
#include <ranges>

#include <cassert>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <ios>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <unordered_map>

#include "mpi.h"
#include "pcgnslib.h"
#include "mini/algebra/eigen.hpp"
#include "mini/mesh/cgns.hpp"
#include "mini/coordinate/face.hpp"
#include "mini/integrator/face.hpp"
#include "mini/coordinate/cell.hpp"
#include "mini/integrator/cell.hpp"
#include "mini/polynomial/concept.hpp"

namespace mini {
namespace mesh {
namespace part {

/**
 * @brief Mimic CGNS's `GridCoordinates_t`, but partitioned.
 * 
 * @tparam Int  Type of integers.
 * @tparam Scalar  Type of scalars.
 */
template <std::integral Int, std::floating_point Scalar>
struct Coordinates {
  Int head_, size_;
  cgns::ShiftedVector<Int> metis_id_;
  cgns::ShiftedVector<Scalar> x_, y_, z_;
  cgsize_t zone_size_[3][1];
  char zone_name_[33];

  Coordinates(int head, int size)
      : head_(head), size_(size), metis_id_(size, head),
        x_(size, head), y_(size, head), z_(size, head) {
  }
  Coordinates() = default;
  Coordinates(Coordinates const &) = delete;
  Coordinates &operator=(Coordinates const &) = delete;
  Coordinates(Coordinates &&that) noexcept {
    *this = std::move(that);
  }
  Coordinates &operator=(Coordinates &&that) noexcept {
    if (this != &that) {
      head_ = that.head_;
      size_ = that.size_;
      metis_id_ = std::move(that.metis_id_);
      x_ = std::move(that.x_);
      y_ = std::move(that.y_);
      z_ = std::move(that.z_);
      std::memcpy(zone_size_, that.zone_size_, 3 * sizeof(cgsize_t));
      std::memcpy(zone_name_, that.zone_name_, 33);
    }
    return *this;
  }
  ~Coordinates() noexcept = default;

  Int size() const {
    return size_;
  }
  Int head() const {
    return head_;
  }
  Int tail() const {
    return head_ + size_;
  }
  bool has(int i_node) const {
    return head_ <= i_node && i_node < size_ + head_;
  }
};

template <std::integral Int, mini::polynomial::General Polynomial>
struct Part;

template <std::integral Int, mini::polynomial::General Polynomial>
struct Cell;

template <std::integral Int, mini::polynomial::General Polynomial>
struct Face {
  using Scalar = typename Polynomial::Scalar;
  constexpr static int kComponents = Polynomial::K;
  constexpr static int kPhysDim = Polynomial::D;
  using Integrator = integrator::Face<Scalar, kPhysDim>;
  using IntegratorUptr = std::unique_ptr<Integrator>;
  using Coordinate = coordinate::Face<Scalar, kPhysDim>;
  using CoordinateUptr = std::unique_ptr<Coordinate>;
  using Cell = part::Cell<Int, Polynomial>;
  using Global = typename Cell::Global;

 private:
  CoordinateUptr coordinate_ptr_;
  IntegratorUptr integrator_ptr_;
  Cell *holder_, *sharer_;
  Scalar holder_height_, sharer_height_;
  Global holder_to_sharer_;
  Int id_;  // 0-based, local first, then ghost, then boundary

 public:
  Face(CoordinateUptr &&coordinate_ptr, IntegratorUptr &&integrator_ptr,
      Cell *holder, Cell *sharer, Int id)
      : coordinate_ptr_(std::move(coordinate_ptr)),
        integrator_ptr_(std::move(integrator_ptr)),
        holder_(holder), sharer_(sharer),
        holder_height_(holder ? holder->volume() / area() : -1),
        sharer_height_(sharer ? sharer->volume() / area() : -1),
        holder_to_sharer_(-holder->center()),
        id_(id) {
    if (sharer) {
      holder_to_sharer_ += sharer->center();
    } else {
      holder_to_sharer_ += center();
      holder_to_sharer_ *= 2;
    }
  }
  Face(const Face &) = delete;
  Face &operator=(const Face &) = delete;
  Face(Face &&) noexcept = default;
  Face &operator=(Face &&) noexcept = default;
  ~Face() noexcept = default;

  Integrator const &integrator() const {
    assert(integrator_ptr_);
    return *integrator_ptr_;
  }
  Coordinate const &coordinate() const {
    assert(coordinate_ptr_);
    return *coordinate_ptr_;
  }
  Global center() const {
    return integrator().center();
  }
  Scalar area() const {
    return integrator().area();
  }
  Scalar holder_height() const {
    assert(holder_);
    return holder_height_;
  }
  Scalar sharer_height() const {
    assert(sharer_);
    return sharer_height_;
  }
  Int id() const {
    return id_;
  }
  Cell const &holder() const {
    assert(holder_);
    return *holder_;
  }
  Cell *holder_ptr() {
    assert(holder_);
    return holder_;
  }
  Cell const &sharer() const {
    assert(sharer_);
    return *sharer_;
  }
  Global const &HolderToSharer() const {
    return holder_to_sharer_;
  }
  Cell const *other(Cell const *cell) const {
    assert(cell == sharer_ || cell == holder_);
    return cell == holder_ ? sharer_ : holder_;
  }
};

template <std::integral Int, mini::polynomial::General Poly>
struct Cell {
  using Polynomial = Poly;
  using PolynomialUptr = std::unique_ptr<Polynomial>;
  using Scalar = typename Polynomial::Scalar;
  using Integrator = integrator::Cell<Scalar>;
  using IntegratorUptr = std::unique_ptr<Integrator>;
  using Coordinate = coordinate::Cell<Scalar>;
  using CoordinateUptr = std::unique_ptr<Coordinate>;
  using Local = typename Polynomial::Local;
  using Global = typename Polynomial::Global;
  using Value = typename Polynomial::Value;
  using Coeff = typename Polynomial::Coeff;
  static constexpr int K = Polynomial::K;  // number of functions
  static constexpr int N = Polynomial::N;  // size of the basis
  static constexpr int P = Polynomial::P;  // degree of the basis
  static constexpr int D = Polynomial::D;  // dimension of the physical space

  static constexpr int kFields = K * N;
  using Face = part::Face<Int, Polynomial>;

  std::vector<Cell *> adj_cells_;
  std::vector<Face *> adj_faces_;
  std::vector<Face *> boundary_faces_;

 private:
  CoordinateUptr coordinate_ptr_;
  IntegratorUptr integrator_ptr_;
  PolynomialUptr polynomial_ptr_;
  Scalar length_;

 public:
  Int metis_id{-1};

 private:
  friend Part<Int, Poly>;
  Int id_{-1};
  bool inner_ = true;

 public:
  Cell(CoordinateUptr &&coordinate_ptr, IntegratorUptr &&integrator_ptr, Int m_cell)
      : coordinate_ptr_(std::move(coordinate_ptr)),
        integrator_ptr_(std::move(integrator_ptr)),
        polynomial_ptr_(std::make_unique<Polynomial>(*integrator_ptr_)),
        length_(std::cbrt(volume()) * 0.5), metis_id(m_cell) {
  }
  Cell() = default;
  Cell(Cell const &) = delete;
  Cell &operator=(Cell const &) = delete;
  Cell(Cell &&that) noexcept = default;
  Cell &operator=(Cell &&that) noexcept = default;
  ~Cell() noexcept = default;

  Scalar length() const {
    return length_;
  }
  Scalar volume() const {
    return integrator_ptr_->volume();
  }
  Int id() const {
    return id_;
  }
  bool inner() const {
    return inner_;
  }
  bool inter() const {
    return !inner_;
  }
  Global const &center() const {
    return integrator().center();
  }
  Integrator const &integrator() const {
    assert(integrator_ptr_);
    return *integrator_ptr_;
  }
  Coordinate const &coordinate() const {
    assert(coordinate_ptr_);
    return *coordinate_ptr_;
  }
  Polynomial const &polynomial() const {
    return *polynomial_ptr_;
  }
  Polynomial &polynomial() {
    return *polynomial_ptr_;
  }
  Global LocalToGlobal(const Local &local) const {
    return coordinate().LocalToGlobal(local);
  }
  Value GlobalToValue(const Global &global) const {
    return polynomial().GlobalToValue(global);
  }
  int CountCorners() const {
    return coordinate().CountCorners();
  }
  int CountFields() const {
    return polynomial().coeff().cols() * polynomial().coeff().rows();
  }
  auto GlobalToBasisValues(const Global &global) const {
    return polynomial().GlobalToBasisValues(global);
  }

  template <class Callable>
  void Approximate(Callable &&func) {
    polynomial().Approximate(std::forward<Callable>(func));
  }
};

/**
 * @brief Mimic CGNS's `Elements_t`, but partitioned.
 * 
 * @tparam Int  Type of integers.
 * @tparam Polynomial  Type of the approximation on each Cell.
 */
template <std::integral Int, mini::polynomial::General Polynomial>
class Section {
  using Cell = part::Cell<Int, Polynomial>;
  using Scalar = typename Cell::Scalar;

  cgns::ShiftedVector<Cell> cells_;

  // fields_[i_field][i_cell]
  cgns::ShiftedVector<cgns::ShiftedVector<Scalar>> fields_;

  Int head_, size_;
  int npe_;

 public:
  static constexpr int kFields = Cell::kFields;

  Section(int head, int size, int npe)
      : cells_(size, head), fields_(kFields, 1),
        head_(head), size_(size), npe_(npe) {
    for (int i = 1; i <= kFields; ++i) {
      fields_[i] = cgns::ShiftedVector<Scalar>(size, head);
    }
  }
  Section() = default;
  Section(Section const &) = delete;
  Section(Section &&) noexcept = default;
  Section &operator=(Section const &) = delete;
  Section &operator=(Section &&) noexcept = default;
  ~Section() noexcept = default;

  Int head() const {
    return head_;
  }
  Int size() const {
    return size_;
  }
  Int tail() const {
    return head_ + size_;
  }
  int npe() const {
    return npe_;
  }
  bool has(int i_cell) const {
    return head() <= i_cell && i_cell < tail();
  }
  const Cell &operator[](Int i_cell) const {
    return cells_[i_cell];
  }
  Cell &operator[](Int i_cell) {
    return cells_[i_cell];
  }
  Cell const &at(Int i_cell) const {
    return cells_.at(i_cell);
  }
  Cell &at(Int i_cell) {
    return cells_.at(i_cell);
  }
  auto begin() {
    return cells_.begin();
  }
  auto end() {
    return cells_.end();
  }
  auto begin() const {
    return cells_.begin();
  }
  auto end() const {
    return cells_.end();
  }
  auto cbegin() const {
    return cells_.cbegin();
  }
  auto cend() const {
    return cells_.cend();
  }
  cgns::ShiftedVector<Scalar> const &GetField(Int i_field) const {
    return fields_.at(i_field);
  }
  cgns::ShiftedVector<Scalar> &GetField(Int i_field) {
    return fields_.at(i_field);
  }
  void GatherFields() {
    for (int i_cell = head(); i_cell < tail(); ++i_cell) {
      const auto &polynomial = cells_.at(i_cell).polynomial();
      for (int i_field = 1; i_field <= kFields; ++i_field) {
        fields_.at(i_field).at(i_cell) = polynomial.GetScalar(i_field - 1);
      }
    }
  }
  void ScatterFields() {
    for (int i_cell = head(); i_cell < tail(); ++i_cell) {
      auto &polynomial = cells_.at(i_cell).polynomial();
      polynomial.SetCoeff([this, i_cell](int i_field) -> Scalar {
        return this->fields_.at(i_field + 1).at(i_cell);
      });
    }
  }
};

/**
 * @brief Mimic CGNS's `Base_t`, but partitioned.
 * 
 * @tparam Int  Type of integers.
 * @tparam Poly  Type of the approximation on each Cell.
 */
template <std::integral Int, mini::polynomial::General Poly>
class Part {
 public:
  using Index = Int;
  using Polynomial = Poly;
  using Face = part::Face<Int, Polynomial>;
  using Cell = part::Cell<Int, Polynomial>;
  using Scalar = typename Cell::Scalar;
  using Global = typename Cell::Global;
  using Value = typename Cell::Value;
  constexpr static int kDegrees = Polynomial::P;
  constexpr static int kComponents = Polynomial::K;
  constexpr static int kPhysDim = Polynomial::D;

 private:
  struct Connectivity {
    cgns::ShiftedVector<Int> index;
    std::vector<Int> nodes;
    cgsize_t first, last, local_first, local_last;
    ElementType_t type;
    char name[33];
  };
  using NodeIndex = cgns::NodeIndex<Int>;
  using CellIndex = cgns::CellIndex<Int>;
  using Coordinates = part::Coordinates<Int, Scalar>;
  using Section = part::Section<Int, Poly>;
  static constexpr int kLineWidth = 128;
  static constexpr int kFields = Section::kFields;
  static constexpr int i_base = 1;
  static constexpr int i_grid = 1;
  static constexpr auto kIntType
      = sizeof(Int) == 8 ? CGNS_ENUMV(LongInteger) : CGNS_ENUMV(Integer);
  static constexpr auto kRealType
      = sizeof(Scalar) == 8 ? CGNS_ENUMV(RealDouble) : CGNS_ENUMV(RealSingle);
  static const MPI_Datatype kMpiIntType;
  static const MPI_Datatype kMpiRealType;

 public:
  Part(std::string const &directory, int rank, int size)
      : directory_(directory), cgns_file_(directory + "/shuffled.cgns"),
        rank_(rank), size_(size) {
  }

  /**
   * @brief Build the geometry by reading the mesh from the underlying CGNS file.
   * 
   * It should be called after calling `Part::InstallPrototype`.
   * 
   */
  void BuildGeometry() {
    int i_file;
    if (cgp_open(cgns_file_.c_str(), CG_MODE_READ, &i_file)) {
      cgp_error_exit();
    }
    auto txt_file = directory_ + "/partition/" + std::to_string(rank_) + ".txt";
    auto istrm = std::ifstream(txt_file);
    BuildLocalNodes(istrm, i_file);
    auto [recv_nodes, recv_coords] = ShareGhostNodes(istrm);
    BuildGhostNodes(recv_nodes, recv_coords);
    BuildLocalCells(istrm, i_file);
    auto ghost_adj = BuildAdj(istrm);
    auto recv_cells = ShareGhostCells(ghost_adj);
    auto m_to_recv_cells = BuildGhostCells(ghost_adj, recv_cells);
    FillCellPtrs(ghost_adj);
    AddLocalCellId();
    AddGhostCellId();
    BuildLocalFaces();
    BuildGhostFaces(ghost_adj, recv_cells, m_to_recv_cells);
    BuildBoundaryFaces(istrm, i_file);
    if (cgp_close(i_file)) {
      cgp_error_exit();
    }
  }
  void SetFieldNames(std::array<std::string, kComponents> const &names) {
    field_names_ = names;
  }
  const std::string &GetFieldName(int i) const {
    return field_names_.at(i);
  }
  const std::string &GetDirectoryName() const {
    return directory_;
  }
  int mpi_rank() const {
    return rank_;
  }
  int mpi_size() const {
    return size_;
  }

 private:
  int SolnNameToId(int i_file, int i_base, int i_zone,
      std::string const &name) {
    int n_solns;
    if (cg_nsols(i_file, i_base, i_zone, &n_solns)) {
      cgp_error_exit();
    }
    int i_soln;
    for (i_soln = 1; i_soln <= n_solns; ++i_soln) {
      char soln_name[33];
      GridLocation_t loc;
      if (cg_sol_info(i_file, i_base, i_zone, i_soln, soln_name, &loc)) {
        cgp_error_exit();
      }
      if (soln_name == name) {
        break;
      }
    }
    assert(i_soln <= n_solns);
    return i_soln;
  }
  int FieldNameToId(int i_file, int i_base, int i_zone, int i_soln,
      std::string const &name) {
    int n_fields;
    if (cg_nfields(i_file, i_base, i_zone, i_soln, &n_fields)) {
      cgp_error_exit();
    }
    int i_field;
    for (i_field = 1; i_field <= n_fields; ++i_field) {
      char field_name[33];
      DataType_t data_t;
      if (cg_field_info(i_file, i_base, i_zone, i_soln, i_field,
          &data_t, field_name)) {
        cgp_error_exit();
      }
      if (field_name == name) {
        break;
      }
    }
    assert(i_field <= n_fields);
    return i_field;
  }
  void BuildLocalNodes(std::ifstream &istrm, int i_file) {
    if (cg_base_read(i_file, i_base, base_name_, &cell_dim_, &phys_dim_)) {
      cgp_error_exit();
    }
    char line[kLineWidth];
    istrm.getline(line, kLineWidth); assert(line[0] == '#');
    // node coordinates
    while (istrm.getline(line, kLineWidth) && line[0] != '#') {
      int i_zone, head, tail;
      std::sscanf(line, "%d %d %d", &i_zone, &head, &tail);
      auto node_group = Coordinates(head, tail - head);
      if (cg_zone_read(i_file, i_base, i_zone,
          node_group.zone_name_, node_group.zone_size_[0])) {
        cgp_error_exit();
      }
      cgsize_t range_min[] = { head };
      cgsize_t range_max[] = { tail - 1 };
      if (cgp_coord_read_data(i_file, i_base, i_zone, 1,
          range_min, range_max, node_group.x_.data()) ||
          cgp_coord_read_data(i_file, i_base, i_zone, 2,
          range_min, range_max, node_group.y_.data()) ||
          cgp_coord_read_data(i_file, i_base, i_zone, 3,
          range_min, range_max, node_group.z_.data())) {
        cgp_error_exit();
      }
      cgsize_t mem_dimensions[] = { tail - head };
      cgsize_t mem_range_min[] = { 1 };
      cgsize_t mem_range_max[] = { mem_dimensions[0] };
      int i_sol = SolnNameToId(i_file, i_base, i_zone, "DataOnNodes");
      int i_field = FieldNameToId(i_file, i_base, i_zone, i_sol, "MetisIndex");
      if (cgp_field_general_read_data(i_file, i_base, i_zone, i_sol, i_field,
          range_min, range_max, kIntType,
          1, mem_dimensions, mem_range_min, mem_range_max,
          node_group.metis_id_.data())) {
        cgp_error_exit();
      }
      for (int i_node = head; i_node < tail; ++i_node) {
        auto m_node = node_group.metis_id_[i_node];
        m_to_node_index_.emplace(m_node, NodeIndex(i_zone, i_node));
      }
      local_nodes_[i_zone] = std::move(node_group);
    }
  }
  std::pair<
    std::map<Int, std::vector<Int>>,
    std::vector<std::vector<Scalar>>
  > ShareGhostNodes(std::ifstream &istrm) {
    char line[kLineWidth];
    // send nodes info
    std::map<Int, std::vector<Int>> send_nodes;
    while (istrm.getline(line, kLineWidth) && line[0] != '#') {
      int i_part, m_node;
      std::sscanf(line, "%d %d", &i_part, &m_node);
      send_nodes[i_part].emplace_back(m_node);
    }
    std::vector<MPI_Request> requests;
    std::vector<std::vector<Scalar>> send_bufs;
    for (auto &[i_part, nodes] : send_nodes) {
      auto &coords = send_bufs.emplace_back();
      for (auto m_node : nodes) {
        auto [i_zone, i_node] = m_to_node_index_.at(m_node);
        auto const &coord = GetCoord(i_zone, i_node);
        coords.emplace_back(coord[0]);
        coords.emplace_back(coord[1]);
        coords.emplace_back(coord[2]);
      }
      assert(std::is_sorted(nodes.begin(), nodes.end()));
      int n_reals = 3 * nodes.size();
      int tag = i_part;
      auto &request = requests.emplace_back();
      MPI_Isend(coords.data(), n_reals, kMpiRealType, i_part, tag,
          MPI_COMM_WORLD, &request);
    }
    // recv nodes info
    std::map<Int, std::vector<Int>> recv_nodes;
    while (istrm.getline(line, kLineWidth) && line[0] != '#') {
      int i_part, m_node, i_zone, i_node;
      std::sscanf(line, "%d %d %d %d", &i_part, &m_node, &i_zone, &i_node);
      recv_nodes[i_part].emplace_back(m_node);
      m_to_node_index_.emplace(m_node, NodeIndex(i_zone, i_node));
    }
    std::vector<std::vector<Scalar>> recv_coords;
    for (auto &[i_part, nodes] : recv_nodes) {
      assert(std::is_sorted(nodes.begin(), nodes.end()));
      int n_reals = 3 * nodes.size();
      auto &coords = recv_coords.emplace_back(std::vector<Scalar>(n_reals));
      int tag = rank_;
      auto &request = requests.emplace_back();
      MPI_Irecv(coords.data(), n_reals, kMpiRealType, i_part, tag,
          MPI_COMM_WORLD, &request);
    }
    // wait until all send/recv finish
    std::vector<MPI_Status> statuses(requests.size());
    MPI_Waitall(requests.size(), requests.data(), statuses.data());
    requests.clear();
    return { recv_nodes, recv_coords };
  }
  void BuildGhostNodes(std::map<Int, std::vector<Int>> const &recv_nodes,
      std::vector<std::vector<Scalar>> const &recv_coords) {
    // copy node coordinates from buffer to member
    int i_source = 0;
    for (auto &[i_part, nodes] : recv_nodes) {
      auto *xyz = recv_coords[i_source++].data();
      for (auto m_node : nodes) {
        auto [i_zone, i_node] = m_to_node_index_[m_node];
        ghost_nodes_[i_zone][i_node] = { xyz[0], xyz[1] , xyz[2] };
        xyz += 3;
      }
    }
  }

 private:
  std::unordered_map<int, typename Face::IntegratorUptr> face_prototypes_;
  std::unordered_map<int, typename Cell::IntegratorUptr> cell_prototypes_;

 public:
  void InstallPrototype(int npe,
      typename Face::IntegratorUptr &&face_integrator) {
    face_prototypes_.emplace(npe, std::move(face_integrator));
  }
  void InstallPrototype(int npe,
      typename Cell::IntegratorUptr &&cell_integrator) {
    cell_prototypes_.emplace(npe, std::move(cell_integrator));
  }

 private:
  std::pair< typename Face::CoordinateUptr, typename Face::IntegratorUptr >
  BuildIntegratorForFace(int npe, int i_zone, Int const *i_node_list) const {
    // TODO(PVC): use O(1) indexing
    auto const &prototype = face_prototypes_.at(npe);
    auto coordinate_uptr = prototype->coordinate().Clone();
    assert(coordinate_uptr->CountNodes() == npe);
    for (int i = 0; i < npe; ++i) {
      coordinate_uptr->SetGlobal(i, GetCoord(i_zone, i_node_list[i]));
    }
    coordinate_uptr->BuildCenter();
    auto integrator_uptr = prototype->Clone(*coordinate_uptr);
    return { std::move(coordinate_uptr), std::move(integrator_uptr) };
  }

  std::pair< typename Cell::CoordinateUptr, typename Cell::IntegratorUptr >
  BuildIntegratorForCell(int npe, int i_zone, Int const *i_node_list) const {
    // TODO(PVC): use O(1) indexing
    auto const &prototype = cell_prototypes_.at(npe);
    auto coordinate_uptr = prototype->coordinate().Clone();
    assert(coordinate_uptr->CountNodes() == npe);
    for (int i = 0; i < npe; ++i) {
      coordinate_uptr->SetGlobal(i, GetCoord(i_zone, i_node_list[i]));
    }
    coordinate_uptr->BuildCenter();
    auto integrator_uptr = prototype->Clone(*coordinate_uptr);
    return { std::move(coordinate_uptr), std::move(integrator_uptr) };
  }

  void BuildLocalCells(std::ifstream &istrm, int i_file) {
    char line[kLineWidth];
    // build local cells
    while (istrm.getline(line, kLineWidth) && line[0] != '#') {
      int i_zone, i_sect, head, tail;
      std::sscanf(line, "%d %d %d %d", &i_zone, &i_sect, &head, &tail);
      cgsize_t range_min[] = { head };
      cgsize_t range_max[] = { tail - 1 };
      cgsize_t mem_dimensions[] = { tail - head };
      cgsize_t mem_range_min[] = { 1 };
      cgsize_t mem_range_max[] = { mem_dimensions[0] };
      cgns::ShiftedVector<Int> metis_ids(mem_dimensions[0], head);
      int i_sol = SolnNameToId(i_file, i_base, i_zone, "DataOnCells");
      int i_field = FieldNameToId(i_file, i_base, i_zone, i_sol, "MetisIndex");
      if (cgp_field_general_read_data(i_file, i_base, i_zone, i_sol, i_field,
          range_min, range_max, kIntType,
          1, mem_dimensions, mem_range_min, mem_range_max, metis_ids.data())) {
        cgp_error_exit();
      }
      int x, y;
      auto &conn = connectivities_[i_zone][i_sect];
      auto &index = conn.index;
      auto &nodes = conn.nodes;
      index = cgns::ShiftedVector<Int>(mem_dimensions[0] + 1, head);
      if (cg_section_read(i_file, i_base, i_zone, i_sect,
          conn.name, &conn.type, &conn.first, &conn.last, &x, &y)) {
        cgp_error_exit();
      }
      int npe = cgns::CountNodesByType(conn.type);
      for (int i_cell = head; i_cell < tail; ++i_cell) {
        auto m_cell = metis_ids[i_cell];
        m_to_cell_index_.emplace(m_cell,
            CellIndex(i_zone, i_sect, i_cell, npe));
      }
      // TODO(PVC): only for non-mixed Section
      nodes.resize(npe * mem_dimensions[0]);
      for (int i = 0; i < index.size(); ++i) {
        index.at(head + i) = npe * i;
      }
      conn.local_first = range_min[0];
      conn.local_last = range_max[0];
      if (cgp_elements_read_data(i_file, i_base, i_zone, i_sect,
          range_min[0], range_max[0], nodes.data())) {
        cgp_error_exit();
      }
      auto section = Section(head, tail - head, npe);
      local_cells_[i_zone][i_sect] = std::move(section);
      for (int i_cell = head; i_cell < tail; ++i_cell) {
        auto *i_node_list = &nodes[(i_cell - head) * npe];
        auto [coordinate_uptr, integrator_uptr]
            = BuildIntegratorForCell(npe, i_zone, i_node_list);
        auto cell = Cell(std::move(coordinate_uptr),
            std::move(integrator_uptr), metis_ids[i_cell]);
        local_cells_[i_zone][i_sect][i_cell] = std::move(cell);
      }
    }
  }
  void AddLocalCellId() {
    for (auto &[i_zone, zone] : local_cells_) {
      for (auto &[i_sect, sect] : zone) {
        for (auto &cell : sect) {
          inner_and_inter_cells_[cell.inter()].emplace_back(&cell);
        }
      }
    }
    Int id = 0;
    cell_data_.push_back(0);
    for (auto cell_ptr : GetLocalCellPointers()) {
      cell_ptr->id_ = id++;
      cell_data_.push_back(cell_data_.back() + cell_ptr->CountFields());
    }
    assert(id == CountLocalCells());
    assert(id + 1 == cell_data_.size());
    assert(cell_data_.back() == CountLocalCells() * Cell::kFields);
    assert(CountLocalCells() == std::ranges::distance(GetLocalCells()));
    assert(CountLocalCells() == std::ranges::distance(GetLocalCellPointers()));
    assert(CountLocalCells() == std::ranges::distance(GetInnerCellPointers())
                              + std::ranges::distance(GetInterCellPointers()));
  }
  void AddGhostCellId() {
    Int id = CountLocalCells();
    for (auto &[_, cell] : ghost_cells_) {
      cell.id_ = id++;
    }
    assert(CountGhostCells() + CountLocalCells() == id);
    assert(CountGhostCells() == std::ranges::distance(GetGhostCells()));
  }

 public:
  /**
   * @brief Get the global i_dof of the Cell's 0th local i_dof.
   * 
   * @param i_cell 
   * @return Int 
   */
  Int GetCellDataOffset(Int i_cell) const {
    assert(0 <= i_cell && i_cell <= CountLocalCells());
    return cell_data_[i_cell];
  }
  /**
   * @brief Get the number of `Scalar`s in cell data.
   * 
   * @return Int 
   */
  Int GetCellDataSize() const {
    Int n_scalar = GetCellDataOffset(CountLocalCells());
    assert(n_scalar == Cell::kFields * CountLocalCells());
    return n_scalar;
  }

 private:
  struct GhostAdj {
    std::map<Int, std::map<Int, Int>>
        send_npes, recv_npes;  // [i_part][m_cell] -> npe
    std::vector<std::pair<Int, Int>>
        m_cell_pairs;
  };
  GhostAdj BuildAdj(std::ifstream &istrm) {
    char line[kLineWidth];
    // local adjacency
    while (istrm.getline(line, kLineWidth) && line[0] != '#') {
      int i, j;
      std::sscanf(line, "%d %d", &i, &j);
      local_adjs_.emplace_back(i, j);
    }
    // ghost adjacency
    auto ghost_adj = GhostAdj();
    auto &send_npes = ghost_adj.send_npes;
    auto &recv_npes = ghost_adj.recv_npes;
    auto &m_cell_pairs = ghost_adj.m_cell_pairs;
    while (istrm.getline(line, kLineWidth) && line[0] != '#') {
      int p, i, j, npe_i, npe_j;
      std::sscanf(line, "%d %d %d %d %d", &p, &i, &j, &npe_i, &npe_j);
      send_npes[p][i] = npe_i;
      recv_npes[p][j] = npe_j;
      m_cell_pairs.emplace_back(i, j);
    }
    return ghost_adj;
  }
  auto ShareGhostCells(GhostAdj const &ghost_adj) {
    auto &send_npes = ghost_adj.send_npes;
    auto &recv_npes = ghost_adj.recv_npes;
    // send cell.i_zone and cell.node_id_list
    std::vector<std::vector<Int>> send_cells;
    std::vector<MPI_Request> requests;
    for (auto &[i_part, npes] : send_npes) {
      auto &send_buf = send_cells.emplace_back();
      for (auto [m_cell, npe] : npes) {
        auto [i_zone, i_sect, i_cell, n_node] = m_to_cell_index_[m_cell];
        assert(npe == n_node);
        auto &conn = connectivities_.at(i_zone).at(i_sect);
        auto &index = conn.index;
        auto &nodes = conn.nodes;
        auto *i_node_list = &(nodes[index[i_cell]]);
        send_buf.emplace_back(i_zone);
        for (int i = 0; i < npe; ++i) {
          send_buf.emplace_back(i_node_list[i]);
        }
      }
      int n_ints = send_buf.size();
      int tag = i_part;
      auto &request = requests.emplace_back();
      MPI_Isend(send_buf.data(), n_ints, kMpiIntType, i_part, tag,
          MPI_COMM_WORLD, &request);
    }
    // recv cell.i_zone and cell.node_id_list
    std::vector<std::vector<Int>> recv_cells;
    for (auto &[i_part, npes] : recv_npes) {
      auto &recv_buf = recv_cells.emplace_back();
      int n_ints = 0;
      for (auto [m_cell, npe] : npes) {
        ++n_ints;
        n_ints += npe;
      }
      int tag = rank_;
      recv_buf.resize(n_ints);
      auto &request = requests.emplace_back();
      MPI_Irecv(recv_buf.data(), n_ints, kMpiIntType, i_part, tag,
          MPI_COMM_WORLD, &request);
    }
    // wait until all send/recv finish
    std::vector<MPI_Status> statuses(requests.size());
    MPI_Waitall(requests.size(), requests.data(), statuses.data());
    requests.clear();
    return recv_cells;
  }
  struct GhostCellIndex {
    int source, head, n_node;
  };
  std::unordered_map<Int, GhostCellIndex> BuildGhostCells(
      GhostAdj const &ghost_adj,
      std::vector<std::vector<Int>> const &recv_cells) {
    auto &recv_npes = ghost_adj.recv_npes;
    // build ghost cells
    std::unordered_map<Int, GhostCellIndex> m_to_recv_cells;
    int i_source = 0;
    for (auto &[i_part, npes] : recv_npes) {
      auto &recv_buf = recv_cells.at(i_source);
      int index = 0;
      for (auto [m_cell, npe] : npes) {
        m_to_recv_cells.emplace(m_cell,
            GhostCellIndex(i_source, index + 1, npe));
        int i_zone = recv_buf[index++];
        auto *i_node_list = &recv_buf[index];
        auto [coordinate_uptr, integrator_uptr]
            = BuildIntegratorForCell(npe, i_zone, i_node_list);
        auto cell = Cell(std::move(coordinate_uptr),
            std::move(integrator_uptr), m_cell);
        ghost_cells_[m_cell] = std::move(cell);
        index += npe;
      }
      ++i_source;
    }
    return m_to_recv_cells;
  }
  void FillCellPtrs(GhostAdj const &ghost_adj) {
    // fill `send_cell_ptrs_`
    for (auto &[i_part, npes] : ghost_adj.send_npes) {
      auto &curr_part = send_cell_ptrs_[i_part];
      assert(curr_part.empty());
      for (auto [m_cell, npe] : npes) {
        auto [i_zone, i_sect, i_cell, n_node] = m_to_cell_index_[m_cell];
        assert(npe == n_node);
        auto &cell = local_cells_.at(i_zone).at(i_sect)[i_cell];
        cell.inner_ = false;
        curr_part.emplace_back(&cell);
      }
    }
    // fill `recv_cell_ptrs_`
    for (auto &[i_part, npes] : ghost_adj.recv_npes) {
      auto &curr_part = recv_cell_ptrs_[i_part];
      assert(curr_part.empty());
      for (auto [m_cell, npe] : npes) {
        auto &cell = ghost_cells_.at(m_cell);
        curr_part.emplace_back(&cell);
      }
    }
    InitializeRequestsAndBuffers(kFields,
        &requests_, &send_coeffs_, &recv_coeffs_);
    assert(send_cell_ptrs_.size() == send_coeffs_.size());
    assert(recv_cell_ptrs_.size() == recv_coeffs_.size());
    assert(requests_.size() == send_coeffs_.size() + recv_coeffs_.size());
  }
  void BuildLocalFaces() {
    // build local faces
    for (auto [m_holder, m_sharer] : local_adjs_) {
      auto &holder_info = m_to_cell_index_[m_holder];
      auto &sharer_info = m_to_cell_index_[m_sharer];
      auto i_zone = holder_info.i_zone;
      // find the common nodes of the holder and the sharer
      auto i_node_cnt = std::unordered_map<Int, Int>();
      auto &conn_i_zone = connectivities_.at(i_zone);
      auto &holder_conn = conn_i_zone.at(holder_info.i_sect);
      auto &sharer_conn = conn_i_zone.at(sharer_info.i_sect);
      auto &holder_nodes = holder_conn.nodes;
      auto &sharer_nodes = sharer_conn.nodes;
      auto holder_head = holder_conn.index[holder_info.i_cell];
      auto sharer_head = sharer_conn.index[sharer_info.i_cell];
      for (int i = 0; i < holder_info.n_node; ++i)
        ++i_node_cnt[holder_nodes[holder_head + i]];
      for (int i = 0; i < sharer_info.n_node; ++i)
        ++i_node_cnt[sharer_nodes[sharer_head + i]];
      auto common_nodes = std::vector<Int>();
      common_nodes.reserve(9/* at most 9 nodes on a Face */);
      for (auto [i_node, cnt] : i_node_cnt)
        if (cnt == 2)
          common_nodes.emplace_back(i_node);
      int face_npe = common_nodes.size();
      // let the normal vector point from holder to sharer
      // see http://cgns.github.io/CGNS_docs_current/sids/conv.figs/hexa_8.png
      auto &zone = local_cells_[i_zone];
      auto &holder = zone[holder_info.i_sect][holder_info.i_cell];
      auto &sharer = zone[sharer_info.i_sect][sharer_info.i_cell];
      holder.adj_cells_.emplace_back(&sharer);
      sharer.adj_cells_.emplace_back(&holder);
      auto *face_node_list = common_nodes.data();
      coordinate::SortNodesOnFace(holder.coordinate(), &holder_nodes[holder_head],
          face_node_list, face_npe);
      auto [coordinate_uptr, integrator_uptr]
          = BuildIntegratorForFace(face_npe, i_zone, face_node_list);
      auto face_uptr = std::make_unique<Face>(std::move(coordinate_uptr),
          std::move(integrator_uptr), &holder, &sharer, local_faces_.size());
      holder.adj_faces_.emplace_back(face_uptr.get());
      sharer.adj_faces_.emplace_back(face_uptr.get());
      local_faces_.emplace_back(std::move(face_uptr));
    }
  }
  void BuildGhostFaces(GhostAdj const &ghost_adj,
      std::vector<std::vector<Int>> const &recv_cells,
      std::unordered_map<Int, GhostCellIndex> const &m_to_recv_cells) {
    auto &m_cell_pairs = ghost_adj.m_cell_pairs;
    // build ghost faces
    for (auto [m_holder, m_sharer] : m_cell_pairs) {
      auto &holder_info = m_to_cell_index_[m_holder];
      auto &sharer_info = m_to_recv_cells.at(m_sharer);
      auto i_zone = holder_info.i_zone;
      // find the common nodes of the holder and the sharer
      auto i_node_cnt = std::unordered_map<Int, Int>();
      auto &holder_conn = connectivities_.at(i_zone).at(holder_info.i_sect);
      auto &holder_nodes = holder_conn.nodes;
      auto &sharer_nodes = recv_cells[sharer_info.source];
      auto holder_head = holder_conn.index[holder_info.i_cell];
      auto sharer_head = sharer_info.head;
      for (int i = 0; i < holder_info.n_node; ++i)
        ++i_node_cnt[holder_nodes[holder_head + i]];
      for (int i = 0; i < sharer_info.n_node; ++i)
        ++i_node_cnt[sharer_nodes[sharer_head + i]];
      auto common_nodes = std::vector<Int>();
      common_nodes.reserve(4);
      for (auto [i_node, cnt] : i_node_cnt) {
        if (cnt == 2)
          common_nodes.emplace_back(i_node);
      }
      int face_npe = common_nodes.size();
      // let the normal vector point from holder to sharer
      auto &zone = local_cells_[i_zone];
      auto &holder = zone[holder_info.i_sect][holder_info.i_cell];
      auto &sharer = ghost_cells_.at(m_sharer);
      holder.adj_cells_.emplace_back(&sharer);
      auto *face_node_list = common_nodes.data();
      coordinate::SortNodesOnFace(holder.coordinate(), &holder_nodes[holder_head],
          face_node_list, face_npe);
      auto [coordinate_uptr, integrator_uptr]
          = BuildIntegratorForFace(face_npe, i_zone, face_node_list);
      auto face_uptr = std::make_unique<Face>(std::move(coordinate_uptr),
          std::move(integrator_uptr), &holder, &sharer,
          local_faces_.size() + ghost_faces_.size());
      holder.adj_faces_.emplace_back(face_uptr.get());
      ghost_faces_.emplace_back(std::move(face_uptr));
    }
  }

 public:
  template <class Callable>
  Value MeasureL1Error(Callable &&exact_solution, Scalar t_curr) const {
    Value l1_error; l1_error.setZero();
    for (Cell const &cell : GetLocalCells()) {
      auto &integrator = cell.integrator();
      auto &polynomial = cell.polynomial();
      for (int q = 0, n = integrator.CountPoints(); q < n; ++q) {
        Value value = polynomial.GetValue(q);
        value -= exact_solution(integrator.GetGlobal(q), t_curr);
        value = value.cwiseAbs() * integrator.GetGlobalWeight(q);
        l1_error += value;
      }
    }
    return l1_error;
  }

  Int CountLocalCells() const {
    return cell_data_.size() - 1;
  }
  Int CountGhostCells() const {
    return ghost_cells_.size();
  }

  bool IsLocal(Int i_cell) const {
    return i_cell < CountLocalCells();
  }
  bool IsGhost(Int i_cell) const {
    return !IsLocal(i_cell);
  }

  void GatherSolutions() {
    int n_zones = local_nodes_.size();
    for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
      int n_sects = local_cells_[i_zone].size();
      for (int i_sect = 1; i_sect <= n_sects; ++i_sect) {
        local_cells_[i_zone][i_sect].GatherFields();
      }
    }
  }
  void ScatterSolutions() {
    int n_zones = local_nodes_.size();
    for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
      int n_sects = local_cells_[i_zone].size();
      for (int i_sect = 1; i_sect <= n_sects; ++i_sect) {
        local_cells_[i_zone][i_sect].ScatterFields();
      }
    }
  }
  void WriteSolutions(std::string const &soln_name = "0") const {
    int n_zones = local_nodes_.size();
    int i_file, i;
    auto cgns_file = directory_ + "/" + soln_name + ".cgns";
    if (rank_ == 0) {
      if (cg_open(cgns_file.c_str(), CG_MODE_WRITE, &i_file)) {
        cgp_error_exit();
      }
      if (cg_base_write(i_file, base_name_, cell_dim_, phys_dim_, &i)
          || i != i_base) {
        cgp_error_exit();
      }
      for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
        auto &node_group = local_nodes_.at(i_zone);
        if (cg_zone_write(i_file, i_base, node_group.zone_name_,
            node_group.zone_size_[0], CGNS_ENUMV(Unstructured), &i)
            || i != i_zone) {
          cgp_error_exit();
        }
        if (cg_grid_write(i_file, i_base, i_zone, "GridCoordinates", &i)
            || i != i_grid) {
          cgp_error_exit();
        }
      }
      if (cg_close(i_file)) {
        cgp_error_exit();
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (cgp_open(cgns_file.c_str(), CG_MODE_MODIFY, &i_file)) {
      cgp_error_exit();
    }
    for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
      // write node coordinates
      auto &node_group = local_nodes_.at(i_zone);
      cgsize_t range_min[] = { node_group.head() };
      cgsize_t range_max[] = { node_group.tail() - 1 };
      auto data_type = std::is_same_v<Scalar, double> ?
          CGNS_ENUMV(RealDouble) : CGNS_ENUMV(RealSingle);
      int i_coord;
      if (cgp_coord_write(i_file, i_base, i_zone, data_type, "CoordinateX",
          &i_coord) || cgp_coord_write_data(i_file, i_base, i_zone, i_coord,
          range_min, range_max, node_group.x_.data())) {
        cgp_error_exit();
      }
      if (cgp_coord_write(i_file, i_base, i_zone, data_type, "CoordinateY",
          &i_coord) || cgp_coord_write_data(i_file, i_base, i_zone, i_coord,
          range_min, range_max, node_group.y_.data())) {
        cgp_error_exit();
      }
      if (cgp_coord_write(i_file, i_base, i_zone, data_type, "CoordinateZ",
          &i_coord) || cgp_coord_write_data(i_file, i_base, i_zone, i_coord,
          range_min, range_max, node_group.z_.data())) {
        cgp_error_exit();
      }
      int n_sects = connectivities_.at(i_zone).size();
      for (int i_sect = 1; i_sect <= n_sects; ++i_sect) {
        auto &sect = connectivities_.at(i_zone).at(i_sect);
        if (cgp_section_write(i_file, i_base, i_zone, sect.name,
            sect.type, sect.first, sect.last, 0/* n_boundary */, &i)
            || i != i_sect) {
          cgp_error_exit();
        }
        if (cgp_elements_write_data(i_file, i_base, i_zone, i_sect,
            sect.local_first, sect.local_last, sect.nodes.data())) {
          cgp_error_exit();
        }
      }
      int n_solns;
      if (cg_nsols(i_file, i_base, i_zone, &n_solns)) {
        cgp_error_exit();
      }
      int i_soln;
      if (cg_sol_write(i_file, i_base, i_zone, "DataOnCells",
          CGNS_ENUMV(CellCenter), &i_soln)) {
        cgp_error_exit();
      }
      auto &zone = local_cells_.at(i_zone);
      for (int i_field = 1; i_field <= kFields; ++i_field) {
        int n_sects = zone.size();
        for (int i_sect = 1; i_sect <= n_sects; ++i_sect) {
          auto &section = zone.at(i_sect);
          auto field_name = "Field" + std::to_string(i_field);
          int field_id;
          if (cgp_field_write(i_file, i_base, i_zone, i_soln, kRealType,
              field_name.c_str(),  &field_id)) {
            cgp_error_exit();
          }
          assert(field_id == i_field);
          cgsize_t first[] = { section.head() };
          cgsize_t last[] = { section.tail() - 1 };
          if (cgp_field_write_data(i_file, i_base, i_zone, i_soln, i_field,
              first, last, section.GetField(i_field).data())) {
            cgp_error_exit();
          }
        }
      }
    }
    if (cgp_close(i_file)) {
      cgp_error_exit();
    }
  }
  void ReadSolutions(std::string const &soln_name) {
    int n_zones = local_nodes_.size();
    int i_file;
    auto cgns_file = directory_ + "/" + soln_name + ".cgns";
    if (cgp_open(cgns_file.c_str(), CG_MODE_READ, &i_file)) {
      cgp_error_exit();
    }
    for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
      auto &zone = local_cells_.at(i_zone);
      int n_solns;
      if (cg_nsols(i_file, i_base, i_zone, &n_solns)) {
        cgp_error_exit();
      }
      int i_soln = SolnNameToId(i_file, i_base, i_zone, "DataOnCells");
      for (int i_field = 1; i_field <= kFields; ++i_field) {
        int n_sects = zone.size();
        for (int i_sect = 1; i_sect <= n_sects; ++i_sect) {
          auto &section = zone.at(i_sect);
          char field_name[33];
          DataType_t data_type;
          if (cg_field_info(i_file, i_base, i_zone, i_soln, i_field,
              &data_type, field_name)) {
            cgp_error_exit();
          }
          cgsize_t first[] = { section.head() };
          cgsize_t last[] = { section.tail() - 1 };
          if (cgp_field_read_data(i_file, i_base, i_zone, i_soln, i_field,
              first, last, section.GetField(i_field).data())) {
            cgp_error_exit();
          }
        }
      }
    }
    if (cgp_close(i_file)) {
      cgp_error_exit();
    }
  }
  /**
   * @brief Initialize data structures used in ShareGhostCellData and UpdateGhostCellData.
   * 
   * @param n_scalar_per_cell 
   * @param requests_ptr 
   * @param send_data_ptr 
   * @param recv_data_ptr 
   */
  void InitializeRequestsAndBuffers(Int n_scalar_per_cell,
      std::vector<MPI_Request> *requests_ptr,
      std::vector<std::vector<Scalar>> *send_data_ptr,
      std::vector<std::vector<Scalar>> *recv_data_ptr) const {
    assert(requests_ptr->empty());
    assert(send_data_ptr->empty());
    assert(recv_data_ptr->empty());
    for (auto &[i_part, cell_ptrs] : send_cell_ptrs_) {
      send_data_ptr->emplace_back(cell_ptrs.size() * n_scalar_per_cell);
    }
    for (auto &[i_part, cell_ptrs] : recv_cell_ptrs_) {
      recv_data_ptr->emplace_back(cell_ptrs.size() * n_scalar_per_cell);
    }
    requests_ptr->resize(send_data_ptr->size() + recv_data_ptr->size());
  }

  /**
   * @brief Register the sharing of data on `Cell`s between neighboring `Part`s.
   * 
   * @tparam M 
   * @param requests_ptr 
   * @param send_data_ptr 
   * @param recv_data_ptr 
   * @param move_data_from_cell_to_buffer 
   */
  template <class M>
  void ShareGhostCellData(std::vector<MPI_Request> *requests_ptr,
      std::vector<std::vector<Scalar>> *send_data_ptr,
      std::vector<std::vector<Scalar>> *recv_data_ptr,
      M &&move_data_from_cell_to_buffer) {
    int i_req = 0;
    // send Data on ghost Cells of each neighboring Part
    int i_buf = 0;
    for (auto &[i_part, cell_ptrs] : send_cell_ptrs_) {
      auto &send_buf = send_data_ptr->at(i_buf++);
      Scalar *data = send_buf.data();
      for (auto *cell_ptr : cell_ptrs) {
        data = move_data_from_cell_to_buffer(cell_ptr, data);
      }
      assert(data == send_buf.data() + send_buf.size());
      int tag = i_part;
      auto &request = requests_ptr->at(i_req++);
      MPI_Isend(send_buf.data(), send_buf.size(), kMpiRealType, i_part, tag,
          MPI_COMM_WORLD, &request);
    }
    // recv Data on ghost Cells from each neighboring Part
    i_buf = 0;
    for (auto &[i_part, cell_ptrs] : recv_cell_ptrs_) {
      auto &recv_buf = recv_data_ptr->at(i_buf++);
      int tag = rank_;
      auto &request = requests_ptr->at(i_req++);
      MPI_Irecv(recv_buf.data(), recv_buf.size(), kMpiRealType, i_part, tag,
          MPI_COMM_WORLD, &request);
    }
    assert(i_req == send_data_ptr->size() + recv_data_ptr->size());
  }
  void ShareGhostCellCoeffs() {
    auto operation = [](Cell const *cell_ptr, Scalar *data) -> Scalar * {
      return cell_ptr->polynomial().WriteCoeffTo(data);
    };
    ShareGhostCellData(&requests_, &send_coeffs_, &recv_coeffs_, operation);
  }

  /**
   * @brief Finish the sharing of data on `Cell`s between neighboring `Part`s.
   * 
   * @tparam M 
   * @param requests_ptr 
   * @param recv_data_ptr 
   * @param move_data_from_buffer_to_cell 
   */
  template <class M>
  void UpdateGhostCellData(std::vector<MPI_Request> *requests_ptr,
      std::vector<std::vector<Scalar>> *recv_data_ptr,
      M &&move_data_from_buffer_to_cell) {
    // wait until all send/recv finish
    std::vector<MPI_Status> statuses(requests_ptr->size());
    MPI_Waitall(requests_ptr->size(), requests_ptr->data(), statuses.data());
    int req_size = requests_ptr->size();
    requests_ptr->clear();
    requests_ptr->resize(req_size);
    // update coeffs
    int i_buf = 0;
    for (auto &[i_part, cell_ptrs] : recv_cell_ptrs_) {
      auto &recv_buf = recv_data_ptr->at(i_buf++);
      Scalar const *data = recv_buf.data();
      for (auto *cell_ptr : cell_ptrs) {
        data = move_data_from_buffer_to_cell(cell_ptr, data);
      }
      assert(data == recv_buf.data() + recv_buf.size());
    }
  }
  void UpdateGhostCellCoeffs() {
    auto operation = [](Cell *cell_ptr, Scalar const *data) -> Scalar const * {
      return cell_ptr->polynomial().GetCoeffFrom(data);
    };
    UpdateGhostCellData(&requests_, &recv_coeffs_, operation);
  }

  // Viewers of `Cell`s and `Face`s:
  /**
   * @brief Get a range of `(Cell const &)`, which contains all ghost `Cell`s.
   * 
   * @return std::ranges::input_range 
   */
  std::ranges::input_range auto GetGhostCells() const {
    return ghost_cells_ | std::views::values;
  }
  /**
   * @brief Get a range of `(Cell const &)`, which contains all local `Cell`s.
   * 
   * @return std::ranges::input_range 
   */
  std::ranges::input_range auto GetLocalCells() const {
    return local_cells_       // range of pair<Index, Zone>
        | std::views::values  // range of (Zone = range of pair<Index, Section>)
        | std::views::join    // range of pair<Index, Section>
        | std::views::values  // range of (Section = range of Cell)
        | std::views::join;   // range of Cell
  }
  /**
   * @brief Get a range of `(Cell *)`, which contains all local `Cell`s.
   * 
   * @return std::ranges::input_range 
   */
  std::ranges::input_range auto GetLocalCellPointers() {
    return inner_and_inter_cells_ | std::views::join;
  }
  /**
   * @brief Get all local `Cell`s not adjacent to ghost cells.
   * 
   * @return std::vector<Cell *> 
   */
  std::vector<Cell *> const &GetInnerCellPointers() {
    return inner_and_inter_cells_[0];
  }
  /**
   * @brief Get all local `Cell`s adjacent to ghost cells.
   * 
   * @return std::vector<Cell *> 
   */
  std::vector<Cell *> const &GetInterCellPointers() {
    return inner_and_inter_cells_[1];
  }
  /**
   * @brief Get a range of `(const Face &)` for local `Face`s.
   * 
   * @return std::ranges::input_range the range of `Face`s
   */
  std::ranges::input_range auto GetLocalFaces() const {
    auto t = [](auto &uptr) -> const Face & { return *uptr; };
    return local_faces_ | std::views::transform(t);
  }
  /**
   * @brief Get a range of `(const Face &)` for ghost `Face`s.
   * 
   * @return std::ranges::input_range the range of `Face`s
   */
  std::ranges::input_range auto GetGhostFaces() const {
    auto t = [](auto &uptr) -> const Face & { return *uptr; };
    return ghost_faces_ | std::views::transform(t);
  }
  /**
   * @brief Get a range of `(Face *)` for `Face`s on a given `Section`.
   * 
   * @param name the name of the `Section`
   * @return std::ranges::input_range the range of `Face`s
   */
  std::ranges::input_range auto
  GetBoundaryFacePointers(std::string const &name) {
    cgns::ShiftedVector<std::unique_ptr<Face>> &faces
        = *name_to_faces_.at(name);
    auto t = [](auto &uptr) -> Face * { return uptr.get(); };
    return faces | std::views::transform(t);
  }
  /**
   * @brief Get a range of `(const Face &)` for `Face`s on a given `Section`.
   * 
   * @param name the name of the `Section`
   * @return std::ranges::input_range the range of `Face`s
   */
  std::ranges::input_range auto
  GetBoundaryFaces(std::string const &name) const {
    auto const &faces = const_cast<Part *>(this)->GetBoundaryFacePointers(name);
    auto t = [](const Face *ptr) -> const Face & { return *ptr; };
    return faces | std::views::transform(t);
  }
  /**
   * @brief Get a range of `(const Face &)` for `Face`s on boundaries.
   * 
   * @return std::ranges::input_range the range of `Face`s sorted by their id's.
   */
  std::ranges::input_range auto
  GetBoundaryFaces() const {
    auto t = [](auto &uptr) -> const Face & { return *uptr; };
    return bound_faces_
        | std::views::values | std::views::join
        | std::views::values | std::views::join
        | std::views::transform(t);
  }

 private:
  std::map<Int, Coordinates>
      local_nodes_;  // [i_zone] -> a Coordinates obj
  std::unordered_map<Int, std::unordered_map<Int, Global>>
      ghost_nodes_;  // [i_zone][i_node] -> a Global obj
  std::unordered_map<Int, NodeIndex>
      m_to_node_index_;  // [m_node] -> a NodeIndex obj
  std::unordered_map<Int, CellIndex>
      m_to_cell_index_;  // [m_cell] -> a CellIndex obj
  std::map<Int, std::map<Int, Connectivity>>
      connectivities_;  // [i_zone][i_sect] -> a Connectivity obj
  std::map<Int, std::map<Int, Section>>
      local_cells_;  // [i_zone][i_sect][i_cell] -> a Cell obj
  std::array<std::vector<Cell *>, 2>
      inner_and_inter_cells_;  // 0 = inner, 1 = inter
  std::vector<Int>
      cell_data_;  // [i_cell] -> global i_dof of the Cell's 0th local i_dof
  std::map<Int, std::vector<Cell *>>
      send_cell_ptrs_, recv_cell_ptrs_;  // [i_part] -> vector<Cell *>
  std::vector<std::vector<Scalar>>
      send_coeffs_, recv_coeffs_;
  std::unordered_map<Int, Cell>
      ghost_cells_;  // [m_cell] -> a Cell obj
  std::vector<std::pair<Int, Int>>
      local_adjs_;  // [i_pair] -> { m_holder, m_sharer }
  std::vector<std::unique_ptr<Face>>
      local_faces_, ghost_faces_;  // [i_face] -> a uptr of Face
  std::map<Int, std::map<Int, cgns::ShiftedVector<std::unique_ptr<Face>>>>
      bound_faces_;  // [i_zone][i_sect][i_face] -> a uptr of Face
  std::unordered_map<std::string, cgns::ShiftedVector<std::unique_ptr<Face>> *>
      name_to_faces_;
  std::vector<MPI_Request> requests_;
  std::array<std::string, kComponents> field_names_;
  const std::string directory_;
  const std::string cgns_file_;
  int rank_, size_, cell_dim_, phys_dim_;
  char base_name_[33];

  void BuildBoundaryFaces(std::ifstream &istrm, int i_file) {
    // build a map from (i_zone, i_node) to cells using it
    std::unordered_map<Int, std::unordered_map<Int, std::vector<Int>>>
        z_n_to_m_cells;  // [i_zone][i_node] -> vector of `m_cell`s
    for (auto &[i_zone, zone] : local_cells_) {
      auto &n_to_m_cells = z_n_to_m_cells[i_zone];
      for (auto &[i_sect, sect] : zone) {
        auto &conn = connectivities_.at(i_zone).at(i_sect);
        auto &index = conn.index;
        auto &nodes = conn.nodes;
        for (int i_cell = sect.head(); i_cell < sect.tail(); ++i_cell) {
          auto &cell = sect[i_cell];
          auto m_cell = cell.metis_id;
          for (int i = index.at(i_cell); i < index.at(i_cell+1); ++i) {
            n_to_m_cells[nodes[i]].emplace_back(m_cell);
          }
        }
      }
    }
    char line[kLineWidth];
    Int face_id = local_faces_.size() + ghost_faces_.size();
    // build boundary faces
    std::unordered_map<std::string, std::pair<int, int>>
        name_to_z_s;  // name -> { i_zone, i_sect }
    while (istrm.getline(line, kLineWidth) && line[0] != '#') {
      int i_zone, i_sect, head, tail;
      std::sscanf(line, "%d %d %d %d", &i_zone, &i_sect, &head, &tail);
      auto &faces = bound_faces_[i_zone][i_sect];
      cgsize_t range_min[] = { head };
      cgsize_t range_max[] = { tail - 1 };
      cgsize_t mem_dimensions[] = { tail - head };
      int x, y;
      auto &conn = connectivities_[i_zone][i_sect];
      auto &index = conn.index;
      auto &nodes = conn.nodes;
      if (cg_section_read(i_file, i_base, i_zone, i_sect,
          conn.name, &conn.type, &conn.first, &conn.last, &x, &y)) {
        cgp_error_exit();
      }
      name_to_z_s[conn.name] = { i_zone, i_sect };
      int npe = cgns::CountNodesByType(conn.type);
      nodes = std::vector<Int>(npe * mem_dimensions[0]);
      index = cgns::ShiftedVector<Int>(mem_dimensions[0] + 1, head);
      for (int i = 0; i < index.size(); ++i) {
        index.at(head + i) = npe * i;
      }
      conn.local_first = range_min[0];
      conn.local_last = range_max[0];
      if (cgp_elements_read_data(i_file, i_base, i_zone, i_sect,
          range_min[0], range_max[0], nodes.data())) {
        cgp_error_exit();
      }
      auto &n_to_m_cells = z_n_to_m_cells.at(i_zone);
      for (int i_face = head; i_face < tail; ++i_face) {
        auto *face_node_list = &nodes[(i_face - head) * npe];
        auto cell_cnt = std::unordered_map<int, int>();
        for (int i = index.at(i_face); i < index.at(i_face+1); ++i) {
          for (auto m_cell : n_to_m_cells[nodes[i]]) {
            cell_cnt[m_cell]++;
          }
        }
        Cell *holder_ptr;
        for (auto [m_cell, cnt] : cell_cnt) {
          assert(cnt <= npe);
          if (cnt == npe) {  // this cell holds this face
            auto [z, s, c, n] = m_to_cell_index_[m_cell];
            holder_ptr = &(local_cells_.at(z).at(s).at(c));
            assert(n == holder_ptr->coordinate().CountNodes());
            auto &holder_conn = connectivities_.at(z).at(s);
            auto &holder_nodes = holder_conn.nodes;
            auto holder_head = holder_conn.index[c];
            coordinate::SortNodesOnFace(holder_ptr->coordinate(),
                &holder_nodes[holder_head], face_node_list, npe);
            break;
          }
        }
        auto [coordinate_uptr, integrator_uptr]
            = BuildIntegratorForFace(npe, i_zone, face_node_list);
        auto face_uptr = std::make_unique<Face>(std::move(coordinate_uptr),
            std::move(integrator_uptr), holder_ptr, nullptr, face_id++);
        // the face's normal vector always point from holder to the exterior
        assert((face_uptr->center() - holder_ptr->center()).dot(
            face_uptr->integrator().GetNormalFrame(0)[0]) > 0);
        holder_ptr->boundary_faces_.emplace_back(face_uptr.get());
        faces.emplace_back(std::move(face_uptr));
      }
    }
    // build name to cgns::ShiftedVector of faces
    for (auto &[name, z_s] : name_to_z_s) {
      auto [i_zone, i_sect] = z_s;
      auto &faces = bound_faces_.at(i_zone).at(i_sect);
      name_to_faces_[name] = &faces;
    }
  }

  Global GetCoord(int i_zone, int i_node) const {
    Global coord;
    auto iter_zone = local_nodes_.find(i_zone);
    if (iter_zone != local_nodes_.end() && iter_zone->second.has(i_node)) {
      coord[0] = iter_zone->second.x_[i_node];
      coord[1] = iter_zone->second.y_[i_node];
      coord[2] = iter_zone->second.z_[i_node];
    } else {
      coord = ghost_nodes_.at(i_zone).at(i_node);
    }
    return coord;
  }

 public:
  std::ofstream GetFileStream(std::string const &soln_name, bool binary,
      std::string const &suffix) const {
    char temp[1024];
    if (rank_ == 0) {
      std::snprintf(temp, sizeof(temp), "mkdir -p %s/%s",
          directory_.c_str(), soln_name.c_str());
      if (std::system(temp))
        throw std::runtime_error(temp + std::string(" failed."));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    std::snprintf(temp, sizeof(temp), "%s/%s/%d.%s",
        directory_.c_str(), soln_name.c_str(), rank_, suffix.c_str());
    return std::ofstream(temp,
        std::ios::out | (binary ? (std::ios::binary) : std::ios::out));
  }
};

template <std::integral Int, mini::polynomial::General Polynomial>
MPI_Datatype const Part<Int, Polynomial>::kMpiIntType
    = sizeof(Int) == 8 ? MPI_LONG : MPI_INT;

template <std::integral Int, mini::polynomial::General Polynomial>
MPI_Datatype const Part<Int, Polynomial>::kMpiRealType
    = sizeof(Scalar) == 8 ? MPI_DOUBLE : MPI_FLOAT;

}  // namespace part
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_PART_HPP_
