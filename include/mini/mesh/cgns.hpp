// Copyright 2021 PEI Weicheng and YANG Minghao and JIANG Yuyan
#ifndef MINI_MESH_CGNS_HPP_
#define MINI_MESH_CGNS_HPP_

#include <concepts>
#include <ranges>

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cgnslib.h"

#include "mini/geometry/pi.hpp"

namespace mini {
namespace mesh {

/**
 * @brief Wrappers of APIs and types in [CGNS/MLL](http://cgns.github.io/CGNS_docs_current/midlevel/index.html).
 * 
 */
namespace cgns {

using ElementType = CGNS_ENUMT(ElementType_t);
using PointSetType = CGNS_ENUMT(PointSetType_t);
using BCType = CGNS_ENUMT(BCType_t);
using GridLocation = CGNS_ENUMT(GridLocation_t);
using DataType = CGNS_ENUMT(DataType_t);

int CountNodesByType(ElementType type) {
  switch (type) {
  case CGNS_ENUMV(TRI_3): return 3;
  case CGNS_ENUMV(TRI_6): return 6;
  case CGNS_ENUMV(QUAD_4): return 4;
  case CGNS_ENUMV(QUAD_8): return 8;
  case CGNS_ENUMV(QUAD_9): return 9;
  case CGNS_ENUMV(TETRA_4): return 4;
  case CGNS_ENUMV(TETRA_10): return 10;
  case CGNS_ENUMV(PYRA_5): return 5;
  case CGNS_ENUMV(PYRA_13): return 13;
  case CGNS_ENUMV(PYRA_14): return 14;
  case CGNS_ENUMV(HEXA_8): return 8;
  case CGNS_ENUMV(HEXA_20): return 20;
  case CGNS_ENUMV(HEXA_27): return 27;
  case CGNS_ENUMV(PENTA_6): return 6;
  case CGNS_ENUMV(PENTA_15): return 15;
  case CGNS_ENUMV(PENTA_18): return 18;
  default:
    int npe; cg_npe(type, &npe); return npe;
  }
  return -1;
}

int CountFacesByType(ElementType type) {
  switch (type) {
  case CGNS_ENUMV(TRI_3):
  case CGNS_ENUMV(TRI_6): return 3;
  case CGNS_ENUMV(QUAD_4):
  case CGNS_ENUMV(QUAD_8):
  case CGNS_ENUMV(QUAD_9): return 4;
  case CGNS_ENUMV(TETRA_4):
  case CGNS_ENUMV(TETRA_10): return 4;
  case CGNS_ENUMV(PYRA_5):
  case CGNS_ENUMV(PYRA_13):
  case CGNS_ENUMV(PYRA_14): return 5;
  case CGNS_ENUMV(HEXA_8):
  case CGNS_ENUMV(HEXA_20):
  case CGNS_ENUMV(HEXA_27): return 6;
  case CGNS_ENUMV(PENTA_6):
  case CGNS_ENUMV(PENTA_15):
  case CGNS_ENUMV(PENTA_18): return 5;
  default:
    int npe; cg_npe(type, &npe); return npe;
  }
  return -1;
}

int dim(ElementType type) {
  switch (type) {
  case CGNS_ENUMV(NODE):
    return 0;
  case CGNS_ENUMV(BAR_2):
    return 1;
  case CGNS_ENUMV(TRI_3):
  case CGNS_ENUMV(QUAD_4):
    return 2;
  case CGNS_ENUMV(TETRA_4):
  case CGNS_ENUMV(PENTA_6):
  case CGNS_ENUMV(HEXA_8):
    return 3;
  default:
    assert(type == CGNS_ENUMV(MIXED));
  }
  return -1;
}

/**
 * @brief Index information of a Node.
 * 
 * @tparam Int  Type of integers.
 */
template <std::integral Int>
struct NodeIndex {
  Int i_zone, i_node;

  NodeIndex(Int zi, Int ni) : i_zone(zi), i_node(ni) {}

  NodeIndex() = default;
  NodeIndex(NodeIndex const &) = default;
  NodeIndex &operator=(NodeIndex const &) = default;
  NodeIndex(NodeIndex &&) noexcept = default;
  NodeIndex &operator=(NodeIndex &&) noexcept = default;
  ~NodeIndex() noexcept = default;
};

/**
 * @brief Index information of a Cell.
 * 
 * @tparam Int  Type of integers.
 */
template <std::integral Int = int>
struct CellIndex {
  Int i_zone, i_sect, i_cell, n_node;

  CellIndex(Int zi, Int si, Int ci, Int ni)
      : i_zone(zi), i_sect(si), i_cell(ci), n_node(ni) {}
  CellIndex(Int zi, Int si, Int ci)
      : i_zone(zi), i_sect(si), i_cell(ci) {}

  CellIndex() = default;
  CellIndex(CellIndex const &) = default;
  CellIndex &operator=(CellIndex const &) = default;
  CellIndex(CellIndex &&) noexcept = default;
  CellIndex &operator=(CellIndex &&) noexcept = default;
  ~CellIndex() noexcept = default;
};

template <std::floating_point Real> class File;
template <std::floating_point Real> class Base;
template <std::floating_point Real> class Zone;
template <std::floating_point Real> class Coordinates;
template <std::floating_point Real> class Section;
template <std::floating_point Real> class ZoneBC;
template <std::floating_point Real> class Solution;

template <std::movable T>
class ShiftedVector : public std::vector<T> {
 private:
  using Base = std::vector<T>;
  using size_type = typename Base::size_type;
  size_type shift_{0};

 public:
  ShiftedVector() = default;
  ShiftedVector(size_type size, size_type shift)
      : Base(size), shift_(shift) {
  }
  ShiftedVector(ShiftedVector const &) = default;
  ShiftedVector(ShiftedVector &&) noexcept = default;
  ShiftedVector &operator=(ShiftedVector const &) = default;
  ShiftedVector &operator=(ShiftedVector &&) noexcept = default;
  ~ShiftedVector() noexcept = default;

  T const &operator[](size_type i) const {
    return this->Base::operator[](i - shift_);
  }
  T const &at(size_type i) const {
    return this->Base::at(i - shift_);
  }
  T &operator[](size_type i) {
    return this->Base::operator[](i - shift_);
  }
  T &at(size_type i) {
    return this->Base::at(i - shift_);
  }
};

/**
 * Wrapper of the `GridCoordinates_t` type.
 */
template <std::floating_point Real>
class Coordinates {
 public:  // Constructors:
  Coordinates(Zone<Real> const &zone, int size)
      : zone_(&zone), x_(size), y_(size), z_(size), name_("GridCoordinates") {
  }

 public:  // Copy Control:
  Coordinates(Coordinates const &) = default;
  Coordinates &operator=(const Coordinates &) = default;
  Coordinates(Coordinates &&) noexcept = default;
  Coordinates &operator=(Coordinates &&) noexcept = default;
  ~Coordinates() noexcept = default;

 public:  // Accessors:
  File<Real> const &file() const {
    return zone_->file();
  }
  Base<Real> const &base() const {
    return zone_->base();
  }
  Zone<Real> const &zone() const {
    return *zone_;
  }
  int CountNodes() const {
    return x_.size();
  }
  std::string const &name() const {
    return name_;
  }
  std::vector<Real> const &x() const {
    return x_;
  }
  std::vector<Real> const &y() const {
    return y_;
  }
  std::vector<Real> const &z() const {
    return z_;
  }
  Real x(cgsize_t i_node/* 1-based */) const {
    return x_[i_node - 1];
  }
  Real y(cgsize_t i_node/* 1-based */) const {
    return y_[i_node - 1];
  }
  Real z(cgsize_t i_node/* 1-based */) const {
    return z_[i_node - 1];
  }
  /**
   * Write coordinates to a given `(file, base, zone)` tuple.
   */
  void Write() const {
    int i_coord;
    auto data_type = std::is_same_v<Real, double> ?
        CGNS_ENUMV(RealDouble) : CGNS_ENUMV(RealSingle);
    cg_coord_write(file().id(), base().id(), zone_->id(),
                   data_type, "CoordinateX", x_.data(), &i_coord);
    cg_coord_write(file().id(), base().id(), zone_->id(),
                   data_type, "CoordinateY", y_.data(), &i_coord);
    cg_coord_write(file().id(), base().id(), zone_->id(),
                   data_type, "CoordinateZ", z_.data(), &i_coord);
  }

 public:  // Mutators:
  std::vector<Real> &x() {
    return x_;
  }
  std::vector<Real> &y() {
    return y_;
  }
  std::vector<Real> &z() {
    return z_;
  }
  Real &x(cgsize_t i_node/* 1-based */) {
    return x_[i_node - 1];
  }
  Real &y(cgsize_t i_node/* 1-based */) {
    return y_[i_node - 1];
  }
  Real &z(cgsize_t i_node/* 1-based */) {
    return z_[i_node - 1];
  }

  void Translate(Real dx, Real dy, Real dz) {
    std::ranges::for_each(x_, [dx](Real &x){ x += dx; });
    std::ranges::for_each(y_, [dy](Real &y){ y += dy; });
    std::ranges::for_each(z_, [dz](Real &z){ z += dz; });
  }

  void Dilate(Real cx, Real cy, Real cz, Real s) {
    std::ranges::for_each(x_, [cx, s](Real &x){ x = cx + s * (x - cx); });
    std::ranges::for_each(y_, [cy, s](Real &y){ y = cy + s * (y - cy); });
    std::ranges::for_each(z_, [cz, s](Real &z){ z = cz + s * (z - cz); });
  }

  void RotateZ(Real ox, Real oy, Real degree) {
    auto [cos, sin] = mini::geometry::CosSin(degree);
    for (int i = 0, n = x_.size(); i < n; ++i) {
      auto dx = x_[i] - ox;
      auto dy = y_[i] - oy;
      auto dx_new = dx * cos - dy * sin;
      auto dy_new = dx * sin + dy * cos;
      x_[i] = ox + dx_new;
      y_[i] = oy + dy_new;
    }
  }

  /**
   * Read coordinates from a given `(file, base, zone)` tuple.
   */
  void Read() {
    // All id's are 1-based when passing to CGNS/MLL.
    cgsize_t first = 1;
    cgsize_t last = CountNodes();
    auto data_type = std::is_same_v<Real, double> ?
        CGNS_ENUMV(RealDouble) : CGNS_ENUMV(RealSingle);
    cg_coord_read(file().id(), base().id(), zone_->id(), "CoordinateX",
                  data_type, &first, &last, x_.data());
    cg_coord_read(file().id(), base().id(), zone_->id(), "CoordinateY",
                  data_type, &first, &last, y_.data());
    cg_coord_read(file().id(), base().id(), zone_->id(), "CoordinateZ",
                  data_type, &first, &last, z_.data());
  }

 private:  // Data Members:
  std::string name_;
  std::vector<Real> x_, y_, z_;
  Zone<Real> const *zone_{nullptr};
  int id_;
};

/**
 * Wrapper of the `Elements_t` type.
 */
template <std::floating_point Real>
class Section {
  friend class Zone<Real>;

  bool mixed() const {
    return type_ == CGNS_ENUMV(MIXED);
  }

 public:  // Constructors:
  Section() = default;
  Section(Zone<Real> const &zone, int i_sect, char const *name,
      cgsize_t first, cgsize_t last, int n_boundary_cells, ElementType type)
      : zone_{&zone}, i_sect_{i_sect}, name_{name}, first_{first}, last_{last},
        n_boundary_cells_{n_boundary_cells}, type_{type} {
    dim_ = cgns::dim(type);
  }

 public:  // Copy Control:
  Section(const Section &) = delete;
  Section &operator=(const Section &) = delete;
  Section(Section &&) noexcept = default;
  Section &operator=(Section &&) noexcept = default;
  ~Section() noexcept = default;

 public:  // Accessors:
  File<Real> const &file() const {
    return zone_->file();
  }
  Base<Real> const &base() const {
    return zone_->base();
  }
  Zone<Real> const &zone() const {
    return *zone_;
  }
  std::string const &name() const {
    return name_;
  }
  int id() const {
    return i_sect_;
  }
  cgsize_t CellIdMin() const { return first_; }
  cgsize_t CellIdMax() const { return last_; }
  cgsize_t CountCells() const { return last_ - first_ + 1; }
  ElementType type() const {
    return type_;
  }
  int CountNodesByType() const {
    return cgns::CountNodesByType(type_);
  }
  int CountFacesByType() const {
    return cgns::CountFacesByType(type_);
  }
  int dim() const {
    return dim_;
  }
  const cgsize_t *GetNodeIdList() const {
    return connectivity_.data();
  }
  cgsize_t *GetNodeIdList() {
    return connectivity_.data();
  }

 private:
  const cgsize_t *_GetNodeIdList(cgsize_t i_row) const {
    auto k = mixed() ? start_offset_.at(i_row) : CountNodesByType() * i_row;
    return &(connectivity_.at(k));
  }
  cgsize_t *_GetNodeIdList(cgsize_t i_row) {
    const cgsize_t *row
        = const_cast<const Section *>(this)->_GetNodeIdList(i_row);
    return const_cast<cgsize_t *>(row);
  }

 public:
  std::ranges::input_range auto GetNodeIdRange(cgsize_t i_cell) const {
    auto first = GetNodeIdList(i_cell);
    auto ptr_view = std::views::iota(first, first + CountNodesByType());
    return std::views::transform(ptr_view, [](auto *ptr){ return *ptr; });
  }
  const cgsize_t *GetNodeIdList(cgsize_t i_cell) const {
    return _GetNodeIdList(i_cell - first_);
  }
  cgsize_t *GetNodeIdList(cgsize_t i_cell) {
    return _GetNodeIdList(i_cell - first_);
  }

  void GetCellCenter(cgsize_t i_cell, Real *x, Real *y, Real *z) const {
    Real cx = 0., cy = 0., cz = 0.;
    auto &coordinates = zone().GetCoordinates();
    for (auto i_node : GetNodeIdRange(i_cell)) {
      cx += coordinates.x(i_node);
      cy += coordinates.y(i_node);
      cz += coordinates.z(i_node);
    }
    auto n_node = CountNodesByType();
    *x = cx / n_node;
    *y = cy / n_node;
    *z = cz / n_node;
  }

  /**
   * @brief Write the connectivity_ (and possibly the start_offset_) from the file.
   */
  void Write() const {
    int i_sect;
    if (mixed()) {
      cg_poly_section_write(file().id(), base().id(), zone().id(),
          name_.c_str(), type(), CellIdMin(), CellIdMax(), 0,
          GetNodeIdList(), start_offset_.data(), &i_sect);
    } else {
      cg_section_write(file().id(), base().id(), zone().id(),
          name_.c_str(), type(), CellIdMin(), CellIdMax(), 0,
          GetNodeIdList(), &i_sect);
    }
    assert(i_sect == i_sect_);
  }

  /**
   * @brief Read the connectivity_ (and possibly the start_offset_) from the file.
   */
  void Read() {
    if (mixed()) {
      cgsize_t data_size;
      cg_ElementDataSize(file().id(), base().id(), zone().id(), id(),
          &data_size);
      start_offset_.resize(CountCells() + 1);
      connectivity_.resize(data_size);
      cg_poly_elements_read(file().id(), base().id(), zone().id(), id(),
          GetNodeIdList(), start_offset_.data(), nullptr/* int *parent_data */);
      for (int i : start_offset_) {
        if (i != start_offset_.back()) {
          auto cell_type = static_cast<ElementType>(connectivity_[i]);
          dim_ = std::max(dim_, cgns::dim(cell_type));
        }
      }
    } else {
      connectivity_.resize(CountCells() * CountNodesByType());
      cg_elements_read(file().id(), base().id(), zone().id(), id(),
          GetNodeIdList(), nullptr/* int *parent_data */);
    }
  }

 private:  // Data Members:
  /*
    if (type() == MIXED) {
      connectivity_[i_cell] = { type, node_1, node_2, ..., node_n } of cell_i;
    } else {
      connectivity_[i_cell] = {       node_1, node_2, ..., node_n } of cell_i;
    }
   */
  std::vector<cgsize_t> connectivity_;
  /*
    if (type() == MIXED) {
      start_offset_[0] = 0;
      start_offset_[i] = start_offset_[i-1] + NPE(type_i) + 1;
    } else {
      start_offset_ = {};
    }
   */
  std::vector<cgsize_t> start_offset_;
  std::string name_;
  Zone<Real> const *zone_{nullptr};
  cgsize_t first_, last_;
  int i_sect_, n_boundary_cells_, dim_;
  ElementType type_;
};

template <std::floating_point Real>
struct BC {
  char name[32];
  cgsize_t ptset[2];
  cgsize_t n_pnts, normal_list_flag, normal_list_size;
  int normal_index, n_mesh;
  PointSetType ptset_type;
  BCType type;
  GridLocation location;
  DataType normal_data_type;
};
template <std::floating_point Real>
class ZoneBC {
  std::vector<BC<Real>> bocos_;
  Zone<Real> const *zone_ptr_;

 public:
  explicit ZoneBC(const Zone<Real> &zone)
      : zone_ptr_(&zone) {
  }
  File<Real> const &file() const {
    return zone_ptr_->file();
  }
  Base<Real> const &base() const {
    return zone_ptr_->base();
  }
  Zone<Real> const &zone() const {
    return *zone_ptr_;
  }

  void Read() {
    int n_bocos;
    cg_nbocos(file().id(), base().id(), zone().id(), &n_bocos);
    bocos_.resize(n_bocos + 1);
    for (int i_boco = 1; i_boco <= n_bocos; ++i_boco) {
      auto &boco = bocos_.at(i_boco);
      cg_boco_info(file().id(), base().id(), zone().id(), i_boco,
          boco.name, &boco.type, &boco.ptset_type, &boco.n_pnts,
          &boco.normal_index, &boco.normal_list_size,
          &boco.normal_data_type, &boco.n_mesh);
      assert(boco.n_pnts == 2);
      cg_boco_read(file().id(), base().id(), zone().id(), i_boco,
          boco.ptset, nullptr);
      std::cout << "Read BC[" << i_boco << "] " << boco.name << ' '
          << boco.ptset[0] << ' ' << boco.ptset[1] << std::endl;
      cg_goto(file().id(), base().id(), "Zone_t", zone().id(),
          "ZoneBC_t", 1, "BC_t", i_boco, "end");
      cg_gridlocation_read(&boco.location);
    }
  }
  void UpdateRanges() {
    int n_bocos = CountBCs();
    for (int i_boco = 1; i_boco <= n_bocos; ++i_boco) {
      auto &boco = bocos_.at(i_boco);
      auto &sect = zone_ptr_->GetSection(boco.name);
      boco.ptset[0] = sect.CellIdMin();
      boco.ptset[1] = sect.CellIdMax();
    }
  }
  void Write() const {
    int n_bocos = CountBCs();
    for (int i_boco = 1; i_boco <= n_bocos; ++i_boco) {
      auto &boco = bocos_.at(i_boco);
      int boco_i;
      cg_boco_write(file().id(), base().id(), zone().id(), boco.name,
          boco.type, boco.ptset_type, boco.n_pnts, boco.ptset, &boco_i);
      assert(boco_i == i_boco);
      cg_goto(file().id(), base().id(), "Zone_t", zone().id(),
          "ZoneBC_t", 1, "BC_t", i_boco, "end");
      cg_boco_gridlocation_write(file().id(), base().id(), zone().id(), i_boco,
          boco.location);
      std::cout << "Write BC[" << i_boco << "] " << boco.name << ' '
          << boco.ptset[0] << ' ' << boco.ptset[1] << std::endl;
    }
  }

 private:
  int CountBCs() const {
    int n_bocos = bocos_.size() - (bocos_.size() > 0);
    return n_bocos;
  }
};

template <std::floating_point Real>
class Field {
 public:  // Constructor:
  Field(Solution<Real> const &solution, int fid, char const *name, int size)
      : solution_{&solution}, i_field_{fid}, name_{name},  data_(size) {
  }

 public:  // Copy control:
  Field(const Field &) = default;
  Field &operator=(const Field &) = default;
  Field(Field &&) noexcept = default;
  Field &operator=(Field &&) noexcept = default;
  ~Field() noexcept = default;

 public:  // Accessors:
  Real const &at(int id) const {
    return data_[id-1];
  }
  int size() const {
    return data_.size();
  }
  std::string const &name() const {
    return name_;
  }
  void Write() const {
    int i_field;
    cg_field_write(solution_->file().id(), solution_->base().id(),
        solution_->zone().id(), solution_->id(), CGNS_ENUMV(RealDouble),
        name_.c_str(), data_.data(), &i_field);
    assert(i_field == i_field_);
  }

 public:  // Mutators:
  Real &at(int id) {
    return data_[id-1];
  }
  Real *data() {
    return data_.data();
  }
  auto begin() {
    return data_.begin();
  }
  auto end() {
    return data_.end();
  }

 private:
  std::vector<Real> data_;
  std::string name_;
  Solution<Real> const *solution_{nullptr};
  int i_field_;
};

template <std::floating_point Real>
class Solution {
 public:  // Constructors:
  Solution(Zone<Real> const &zone, int i_soln, char const *name,
           GridLocation location)
      : zone_(&zone), i_soln_(i_soln), name_(name), location_(location) {
  }

 public:  // Copy Control:
  Solution(const Solution &) = delete;
  Solution &operator=(const Solution &) = delete;
  Solution(Solution &&) noexcept = default;
  Solution &operator=(Solution &&) noexcept = default;
  ~Solution() noexcept = default;

 public:  // Accessors:
  File<Real> const &file() const {
    return zone_->file();
  }
  Base<Real> const &base() const {
    return zone_->base();
  }
  Zone<Real> const &zone() const {
    return *zone_;
  }
  int id() const {
    return i_soln_;
  }
  std::string const &name() const {
    return name_;
  }
  GridLocation localtion() const {
    return location_;
  }
  bool OnNodes() const {
    return location_ == CGNS_ENUMV(Vertex);
  }
  bool OnCells() const {
    return location_ == CGNS_ENUMV(CellCenter);
  }
  Field<Real> const &GetField(int i_field) const {
    return *(fields_.at(i_field-1));
  }
  Field<Real> const &GetField(const std::string &name) const {
    for (auto &field : fields_)
      if (field->name() == name)
        return *field;
    throw std::invalid_argument("There is no field named \"" + name + "\".");
    return *(fields_.back());
  }
  int CountFields() const {
    return fields_.size();
  }
  void Write() const {
    int i_soln;
    cg_sol_write(file().id(), base().id(), zone_->id(),
                 name_.c_str(), location_, &i_soln);
    assert(i_soln == i_soln_);
    for (auto &field : fields_) {
      field->Write();
    }
  }

 public:  // Mutators:
  Field<Real> &GetField(int i_field) {
    return *(fields_.at(i_field-1));
  }
  Field<Real> &AddField(char const *name) {
    assert(OnNodes() || OnCells());
    // find an old one
    for (auto &field_uptr : fields_) {
      if (field_uptr->name() == name) {
        return *field_uptr;
      }
    }
    // build a new one
    int size = OnCells() ? zone_->CountCells() : zone_->CountNodes();
    fields_.emplace_back(std::make_unique<Field<Real>>(
        *this, fields_.size()+1, name, size));
    return *(fields_.back());
  }

 private:
  std::vector<std::unique_ptr<Field<Real>>> fields_;
  std::string name_;
  Zone<Real> const *zone_{nullptr};
  GridLocation location_;
  int i_soln_;
};

template <std::floating_point Real>
class Zone {
 public:  // Constructors:
  /**
   * @brief Construct a new Zone object
   * 
   * @param base the reference to the parent
   * @param zid the id of this zone
   * @param name the name of this zone
   * @param n_cells the number of highest-dim cells in this zone
   * @param n_nodes the number of nodes in this zone
   */
  Zone(Base<Real> const &base, int zid, char const *name,
       cgsize_t n_cells, cgsize_t n_nodes)
      : base_(&base), i_zone_(zid), name_(name), n_cells_(n_cells),
        coordinates_(*this, n_nodes), zone_bc_(*this) {
  }

 public:  // Copy Control:
  Zone(const Zone &) = delete;
  Zone &operator=(const Zone &) = delete;
  Zone(Zone &&) noexcept = default;
  Zone &operator=(Zone &&) noexcept = default;
  ~Zone() noexcept = default;

 public:  // Accessors:
  File<Real> const &file() const {
    return base_->file();
  }
  Base<Real> const &base() const {
    return *base_;
  }
  int id() const {
    return i_zone_;
  }
  const std::string &name() const {
    return name_;
  }
  int CountNodes() const {
    return coordinates_.CountNodes();
  }
  /**
   * @brief 
   * 
   * @return int 
   */
  int CountAllCells() const {
    int total = 0;
    for (auto &section : sections_) {
      total += section->CountCells();
    }
    return total;
  }
  /**
   * @brief 
   * 
   * @return int 
   */
  int CountCells() const {
    return n_cells_;
  }
  int CountCellsByType(ElementType type) const {
    int cell_size{0};
    for (auto &section : sections_) {
      if (section->type_ == type) {
        cell_size += section->CountCells();
      }
    }
    return cell_size;
  }
  int CountSections() const {
    return sections_.size();
  }
  int CountSolutions() const {
    return solutions_.size();
  }
  Coordinates<Real> const &GetCoordinates() const {
    return coordinates_;
  }
  Section<Real> const &GetSection(int i_sect) const {
    return *(sections_.at(i_sect-1));
  }
  Section<Real> const &GetSection(const std::string &name) const {
    // TODO(PVC): use hash table
    for (auto &sect : sections_) {
      int pos = 0;
      if (sect->name().size() > name.size()) {
        pos = sect->name().size() - name.size();
      }
      if (sect->name().substr(pos) == name)
        return *sect;
    }
    throw std::invalid_argument("There is no section named \"" + name + "\".");
    return *(sections_.back());
  }
  Solution<Real> const &GetSolution(int i_soln) const {
    return *(solutions_.at(i_soln-1));
  }
  Solution<Real> const &GetSolution(const std::string &name) const {
    // TODO(PVC): use hash table
    for (auto &soln : solutions_)
      if (soln->name() == name)
        return *soln;
    throw std::invalid_argument("There is no solution named \"" + name + "\".");
    return *(solutions_.back());
  }
  void Write(int min_dim, int max_dim) const {
    int i_zone;
    cgsize_t zone_size[3] = {CountNodes(), CountCells(), 0};
    cg_zone_write(file().id(), base().id(), name_.c_str(), zone_size,
                  CGNS_ENUMV(Unstructured), &i_zone);
    assert(i_zone == i_zone_);
    coordinates_.Write();
    zone_bc_.Write();
    for (auto &section : sections_) {
      if (min_dim <= section->dim() && section->dim() <= max_dim)
        section->Write();
    }
    for (auto &solution : solutions_) {
      solution->Write();
    }
  }
  /**
  * Return true if the cell type is supported and consistent with the given dim.
  */
  static bool CheckTypeDim(ElementType type, int cell_dim) {
    switch (type) {
    case CGNS_ENUMV(TRI_3):
    case CGNS_ENUMV(QUAD_4):
      return cell_dim == 2;
    case CGNS_ENUMV(TETRA_4):
    case CGNS_ENUMV(PENTA_6):
    case CGNS_ENUMV(HEXA_8):
      return cell_dim == 3;
    default:
      return type == CGNS_ENUMV(MIXED);
    }
    return false;
  }

 public:  // Mutators:
  Coordinates<Real> &GetCoordinates() {
    return coordinates_;
  }
  Section<Real> &GetSection(int i_sect) {
    return *(sections_.at(i_sect-1));
  }
  Solution<Real> &GetSolution(int i_soln) {
    return *(solutions_.at(i_soln-1));
  }
  void ReadCoordinates() {
    coordinates_.Read();
  }
  void ReadAllSections() {
    int n_sections;
    cg_nsections(file().id(), base().id(), i_zone_, &n_sections);
    sections_.reserve(n_sections);
    for (int i_sect = 1; i_sect <= n_sections; ++i_sect) {
      char section_name[33];
      ElementType cell_type;
      cgsize_t first, last;
      int n_boundary_cells, parent_flag;
      cg_section_read(file().id(), base().id(), i_zone_, i_sect,
                      section_name, &cell_type, &first, &last,
                      &n_boundary_cells, &parent_flag);
      auto &section = sections_.emplace_back(std::make_unique<Section<Real>>(
          *this, i_sect, section_name, first, last, n_boundary_cells,
          cell_type));
      section->Read();
    }
    SortSectionsByDim();
  }
  void ReadZoneBC() {
    zone_bc_.Read();
  }
  void UpdateSectionRanges() {
    zone_bc_.UpdateRanges();
  }
  void ReadSolutions() {
    int n_solutions;
    cg_nsols(file().id(), base_->id(), i_zone_, &n_solutions);
    solutions_.reserve(n_solutions);
    for (int i_soln = 1; i_soln <= n_solutions; ++i_soln) {
      char sol_name[33];
      GridLocation location;
      cg_sol_info(file().id(), base_->id(), i_zone_,
          i_soln, sol_name, &location);
      auto &solution = solutions_.emplace_back(std::make_unique<Solution<Real>>(
          *this, i_soln, sol_name, location));
      int n_fields;
      cg_nfields(file().id(), base_->id(), i_zone_, i_soln, &n_fields);
      for (int i_field = 1; i_field <= n_fields; ++i_field) {
        DataType datatype;
        char field_name[33];
        cg_field_info(file().id(), base_->id(), i_zone_, i_soln,
                      i_field, &datatype, field_name);
        cgsize_t first{1}, last{1};
        if (location == CGNS_ENUMV(Vertex)) {
          last = CountNodes();
        } else if (location == CGNS_ENUMV(CellCenter)) {
          last = CountCells();
        } else {
          assert(false);
        }
        auto &field = solution->AddField(field_name);
        cg_field_read(file().id(), base_->id(), i_zone_, i_soln, field_name,
                      datatype, &first, &last, &(field.at(1)));
      }
    }
  }
  Solution<Real> &AddSolution(char const *sol_name, GridLocation location) {
    // find an old one
    for (auto &soln_uptr : solutions_) {
      if (soln_uptr->name() == sol_name && soln_uptr->localtion() == location) {
        return *soln_uptr;
      }
    }
    // build a new one
    int i_soln = solutions_.size() + 1;
    return *(solutions_.emplace_back(std::make_unique<Solution<Real>>(
        *this, i_soln, sol_name, location)));
  }

  /**
   * @brief Merge the given sections to a single MIXED section.
   * 
   * @param section_list  Sections to be merged. If it is empty, then all sections are merged.
   */
  void MergeSections(std::initializer_list<int> section_list) {
    auto section_set = std::unordered_set<int>(section_list);
    auto merged_sections = std::vector<std::unique_ptr<Section<Real>>>();
    auto mixed_section = std::make_unique<Section<Real>>(
        *this, 0/* i_sect */, "Mixed"/* name */, 1/* first */, 0/* last */,
        0/* n_boundary_cells */, CGNS_ENUMV(MIXED));
    mixed_section->start_offset_.push_back(0);
    for (auto &old_section : sections_) {
      if (section_set.empty() || section_set.count(old_section->id())) {
        auto type = old_section->type();
        auto n_node = old_section->CountNodesByType();
        mixed_section->dim_ = std::max(mixed_section->dim_, old_section->dim());
        for (auto i_cell = old_section->CellIdMin();
            i_cell <= old_section->CellIdMax(); ++i_cell) {
          auto *row = old_section->GetNodeIdList(i_cell);
          mixed_section->connectivity_.push_back(type);
          for (int i = 0; i < n_node; ++i) {
            mixed_section->connectivity_.push_back(row[i]);
          }
          mixed_section->start_offset_.push_back(
              mixed_section->connectivity_.size());
          mixed_section->last_++;
        }
      } else {  // not in the to-be-merged list
        merged_sections.emplace_back(std::move(old_section));
      }
    }
    if (mixed_section->CountCells()) {
      merged_sections.emplace_back(std::move(mixed_section));
    }
    std::swap(sections_, merged_sections);
    SortSectionsByDim();
  }
  /**
   * @brief Split the given MIXED sections to a few non-MIXED sections.
   * 
   * @param section_list  Sections to be merged. If it is empty, then all MIXED sections are splitted.
   */
  void SplitSections(std::initializer_list<int> section_list) {
    auto section_set = std::unordered_set<int>(section_list);
    auto new_sections = std::vector<std::unique_ptr<Section<Real>>>();
    auto type_to_sections
        = std::unordered_map<ElementType, std::unique_ptr<Section<Real>>>();
    int i_sect = sections_.size();
    for (auto &old_section : sections_) {
      if (old_section->mixed() &&
          (section_set.empty() || section_set.count(old_section->id()))) {
        for (auto i_cell = old_section->CellIdMin();
            i_cell <= old_section->CellIdMax(); ++i_cell) {
          auto *row = old_section->GetNodeIdList(i_cell);
          auto type = static_cast<ElementType>(row[0]);
          auto n_node = cgns::CountNodesByType(type);
          auto iter = type_to_sections.find(type);
          if (iter == type_to_sections.end()) {
            auto uptr = std::make_unique<Section<Real>>(*this, i_sect++,
                (std::to_string(type_to_sections.size())+"Mixed").c_str(),
                1/* first */, 0/* last */, 0/* n_boundary_cells */, type);
            iter = type_to_sections.emplace_hint(iter, type, std::move(uptr));
          }
          auto &section = iter->second;
          for (int i = 1; i <= n_node; ++i) {
            section->connectivity_.push_back(row[i]);
          }
          section->last_++;
        }
      } else {  // not mixed or not in the to-be-splitted list
        new_sections.emplace_back(std::move(old_section));
      }
    }
    for (auto &[type, section] : type_to_sections) {
      new_sections.emplace_back(std::move(section));
    }
    std::swap(sections_, new_sections);
    SortSectionsByDim();
  }

 private:
  std::string name_;
  Coordinates<Real> coordinates_;
  std::vector<std::unique_ptr<Section<Real>>> sections_;
  std::vector<std::unique_ptr<Solution<Real>>> solutions_;
  ZoneBC<Real> zone_bc_;
  Base<Real> const *base_{nullptr};
  cgsize_t n_cells_;
  int i_zone_;

  void SortSectionsByDim() {
    int n = sections_.size();
    auto dim_then_oldid = std::vector<std::pair<int, int>>();
    for (auto &sect : sections_) {
      dim_then_oldid.emplace_back(sect->dim(), sect->id());
    }
    auto order = std::vector<int>(n);
    // TODO(PVC): use std::ranges::view in C++23
    std::iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&dim_then_oldid](int lid, int rid) {
      return dim_then_oldid[lid].first > dim_then_oldid[rid].first || (
          dim_then_oldid[lid].first == dim_then_oldid[rid].first &&
          dim_then_oldid[lid].second < dim_then_oldid[rid].second);
    });
    auto sorted_sections = std::vector<std::unique_ptr<Section<Real>>>(n);
    for (int i = 0; i < n; ++i) {
      std::swap(sorted_sections[i], sections_[order[i]]);
    }
    std::swap(sorted_sections, sections_);
    int i_next = 1;
    for (int i_sect = 1; i_sect <= n; ++i_sect) {
      auto &sect = *sections_[i_sect-1];
      sect.i_sect_ = i_sect;
      auto n_cell = sect.CountCells();
      sect.first_ = i_next;
      sect.last_ = sect.first_ + n_cell - 1;
      assert(n_cell == sect.CountCells());
      i_next += n_cell;
    }
    assert(i_next - 1 == CountAllCells());
  }
};

template <std::floating_point Real>
class Base {
 public:  // Constructors:
  Base(File<Real> const &file, int bid, char const *name,
       int cell_dim, int phys_dim)
      : file_(&file), i_base_(bid), name_(name),
        cell_dim_(cell_dim), phys_dim_(phys_dim) {
  }

 public:  // Copy Control:
  Base(const Base &) = delete;
  Base &operator=(const Base &) = delete;
  Base(Base &&) noexcept = default;
  Base &operator=(Base &&) noexcept = default;
  ~Base() noexcept = default;

 public:  // Accessors:
  File<Real> const &file() const {
    return *file_;
  }
  int id() const {
    return i_base_;
  }
  int GetCellDim() const {
    return cell_dim_;
  }
  int GetPhysDim() const {
    return phys_dim_;
  }
  const std::string &name() const {
    return name_;
  }
  int CountZones() const {
    return zones_.size();
  }
  const Zone<Real> &GetZone(int id) const {
    return *(zones_.at(id-1));
  }
  Zone<Real> const &GetUniqueBase() const {
    return const_cast<Base *>(this)->GetUniqueZone();
  }
  void Write(int min_dim, int max_dim) const {
    int i_base;
    cg_base_write(file_->id(), name_.c_str(), cell_dim_, phys_dim_, &i_base);
    assert(i_base == i_base_);
    for (auto &zone : zones_) {
      zone->Write(min_dim, max_dim);
    }
  }

 public:  // Mutators:
  Zone<Real> &GetZone(int id) {
    return *(zones_.at(id-1));
  }
  Zone<Real> &GetUniqueZone() {
    if (CountZones() != 1) {
      throw std::runtime_error("This method can only be called by a 1-Zone_t CGNSBase_t object.");
    }
    return GetZone(1);
  }
  void ReadZones() {
    int n_zones;
    cg_nzones(file().id(), i_base_, &n_zones);
    zones_.reserve(n_zones);
    for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
      char zone_name[33];
      cgsize_t zone_size[3][1];
      cg_zone_read(file().id(), i_base_, i_zone, zone_name, zone_size[0]);
      auto &zone = zones_.emplace_back(std::make_unique<Zone<Real>>(
          *this, i_zone, zone_name,
          /* n_cells */zone_size[1][0], /* n_nodes */zone_size[0][0]));
      zone->ReadCoordinates();
      zone->ReadAllSections();
      zone->ReadZoneBC();
      zone->ReadSolutions();
    }
  }
  void ReadNodeIdList(int i_file) {
    int n_zones;
    cg_nzones(file().id(), i_base_, &n_zones);
    zones_.reserve(n_zones);
    for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
      char zone_name[33];
      cgsize_t zone_size[3][1];
      cg_zone_read(file().id(), i_base_, i_zone, zone_name, zone_size[0]);
      auto &zone = zones_.emplace_back(std::make_unique<Zone<Real>>(
          *this, i_zone, zone_name,
          /* n_cells */zone_size[1][0], /* n_nodes */zone_size[0][0]));
      zone->ReadSectionsWithDim(cell_dim_);
    }
  }

  void Translate(Real dx, Real dy, Real dz) {
    for (auto &zone : zones_) {
      zone->GetCoordinates().Translate(dx, dy, dz);
    }
  }

  void Dilate(Real cx, Real cy, Real cz, Real s) {
    for (auto &zone : zones_) {
      zone->GetCoordinates().Dilate(cx, cy, cz, s);
    }
  }

  void RotateZ(Real ox, Real oy, Real degree) {
    for (auto &zone : zones_) {
      zone->GetCoordinates().RotateZ(ox, oy, degree);
    }
  }

 private:
  std::vector<std::unique_ptr<Zone<Real>>> zones_;
  std::string name_;
  File<Real> const *file_{nullptr};
  int i_base_, cell_dim_, phys_dim_;
};

template <std::floating_point Real>
class File {
 public:  // Constructors:
  explicit File(const std::string &name)
      : name_(name) {
  }
  File(const std::string &dir, const std::string &name)
      : name_(dir) {
    if (name_.back() != '/')
      name_.push_back('/');
    name_ += name;
  }

 public:  // Copy Control:
  File(const File &) = delete;
  File &operator=(const File &) = delete;
  File(File &&) noexcept = default;
  File &operator=(File &&) noexcept = default;
  ~File() noexcept = default;

 public:  // Accessors:
  int id() const {
    return i_file_;
  }
  std::string const &name() const {
    return name_;
  }
  int CountBases() const {
    return bases_.size();
  }
  Base<Real> const &GetBase(int id) const {
    return *(bases_.at(id-1));
  }
  Base<Real> const &GetUniqueBase() const {
    return const_cast<File *>(this)->GetUniqueBase();
  }

 public:  // Mutators:
  Base<Real> &GetBase(int id) {
    return *(bases_.at(id-1));
  }
  Base<Real> &GetUniqueBase() {
    if (CountBases() != 1) {
      throw std::runtime_error("This method can only be called by a 1-CGNSBase_t CGNSTree_t object.");
    }
    return GetBase(1);
  }
  void ReadBases() {
    if (cg_open(name_.c_str(), CG_MODE_READ, &i_file_)) {
      cg_error_exit();
    }
    bases_.clear();
    int n_bases;
    cg_nbases(i_file_, &n_bases);
    bases_.reserve(n_bases);
    for (int i_base = 1; i_base <= n_bases; ++i_base) {
      char base_name[33];
      int cell_dim{-1}, phys_dim{-1};
      cg_base_read(i_file_, i_base, base_name, &cell_dim, &phys_dim);
      auto &base = bases_.emplace_back(std::make_unique<Base<Real>>(
          *this, i_base, base_name, cell_dim, phys_dim));
      base->ReadZones();
    }
    cg_close(i_file_);
  }
  void ReadNodeIdList() {
    if (cg_open(name_.c_str(), CG_MODE_READ, &i_file_)) {
      cg_error_exit();
    }
    bases_.clear();
    int n_bases;
    cg_nbases(i_file_, &n_bases);
    bases_.reserve(n_bases);
    for (int i_base = 1; i_base <= n_bases; ++i_base) {
      char base_name[33];
      int cell_dim, phys_dim;
      cg_base_read(i_file_, i_base, base_name, &cell_dim, &phys_dim);
      auto &base = bases_.emplace_back(std::make_unique<Base<Real>>(
          *this, i_base, base_name, cell_dim, phys_dim));
      base->ReadNodeIdList(i_file_);
    }
    cg_close(i_file_);
  }
  void Write(const std::string &file_name, int min_dim = 0, int max_dim = 3) {
    name_ = file_name;
    if (cg_open(file_name.c_str(), CG_MODE_WRITE, &i_file_)) {
      cg_error_exit();
    }
    for (auto &base : bases_) {
      base->Write(min_dim, max_dim);
    }
    cg_close(i_file_);
  }

  void Translate(Real dx, Real dy, Real dz) {
    for (auto &base : bases_) {
      base->Translate(dx, dy, dz);
    }
  }

  void Dilate(Real cx, Real cy, Real cz, Real s) {
    for (auto &base : bases_) {
      base->Dilate(cx, cy, cz, s);
    }
  }

  void RotateZ(Real ox, Real oy, Real degree) {
    for (auto &base : bases_) {
      base->RotateZ(ox, oy, degree);
    }
  }

 private:
  std::vector<std::unique_ptr<Base<Real>>> bases_;
  std::string name_;
  int i_file_;
};

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_CGNS_HPP_
