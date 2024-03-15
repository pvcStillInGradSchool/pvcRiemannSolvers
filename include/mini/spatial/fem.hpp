// Copyright 2023 PEI Weicheng
#ifndef MINI_SPATIAL_FEM_HPP_
#define MINI_SPATIAL_FEM_HPP_

#include <cassert>
#include <fstream>
#include <functional>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "mini/riemann/concept.hpp"
#include "mini/temporal/ode.hpp"
#include "mini/constant/index.hpp"

namespace mini {
namespace spatial {

using namespace mini::constant::index;

static bool Near(auto const &x, auto const &y) {
  return (x - y).norm() < 1e-12;
}

template <typename P>
class DummySource {
 public:
  using Part = P;
  using Cell = typename Part::Cell;
  using Scalar = typename Cell::Scalar;

  void UpdateCoeff(const Cell &cell, double t_curr, Scalar *coeff) {
  }
};

template <typename Part>
class FiniteElement : public temporal::System<typename Part::Scalar> {
 public:
  using Riemann = typename Part::Riemann;
  using Scalar = typename Part::Scalar;
  using Face = typename Part::Face;
  using Cell = typename Part::Cell;
  using Index = typename Part::Index;
  using Global = typename Cell::Global;
  using Gauss = typename Cell::Gauss;
  using Projection = typename Cell::Projection;
  using Coeff = typename Projection::Coeff;
  using Value = typename Projection::Value;
  using Temporal = temporal::System<typename Part::Scalar>;
  using Column = typename Temporal::Column;

 protected:
  std::vector<std::string> supersonic_outlet_, inviscid_wall_;
  using Function = std::function<Value(const Global &, double)>;
  std::unordered_map<std::string, Function> supersonic_inlet_,
      subsonic_inlet_, subsonic_outlet_, smart_boundary_, no_slip_wall_;

  Part *part_ptr_;
  double t_curr_;
  size_t cell_data_size_;
#ifdef ENABLE_LOGGING
  std::unique_ptr<std::ofstream> log_;
#endif

 public:
  explicit FiniteElement(Part *part_ptr)
      : part_ptr_(part_ptr), cell_data_size_(part_ptr->GetCellDataSize()) {
    assert(cell_data_size_ == Cell::kFields * part_ptr->CountLocalCells());
#ifdef ENABLE_LOGGING
    log_ = std::make_unique<std::ofstream>();
#endif
  }
  FiniteElement(const FiniteElement &) = default;
  FiniteElement &operator=(const FiniteElement &) = default;
  FiniteElement(FiniteElement &&) noexcept = default;
  FiniteElement &operator=(FiniteElement &&) noexcept = default;
  ~FiniteElement() noexcept {
#ifdef ENABLE_LOGGING
    if (log_->is_open()) {
      log_->close();
    }
#endif
  }

  virtual std::string name() const {
    return "FEM";
  }
  std::string fullname() const {
    return name() + "_" + std::to_string(part_ptr_->mpi_rank());
  }

  Part *part_ptr() {
    return part_ptr_;
  }
  Part const &part() const {
    return *part_ptr_;
  }

#ifdef ENABLE_LOGGING
  std::ofstream &log() const {
    if (!log_->is_open()) {
      log_->open(fullname() + ".txt");
    }
    assert(log_->is_open());
    return *log_;
  }
#endif

  Scalar *AddCellDataOffset(Column *column, Index i_cell) const {
    auto *data = column->data() + part().GetCellDataOffset(i_cell);
    assert(column->data() <= data);
    assert(data + Cell::kFields <= column->data() + column->size());
    return data;
  }

  Scalar const *AddCellDataOffset(Column const &column, Index i_cell) const {
    auto *data = column.data() + part().GetCellDataOffset(i_cell);
    assert(column.data() <= data);
    assert(data + Cell::kFields <= column.data() + column.size());
    return data;
  }

 public:  // set BCs
  template <typename Callable>
  void SetSmartBoundary(const std::string &name, Callable &&func) {
    smart_boundary_[name] = func;
  }
  template <typename Callable>
  void SetSupersonicInlet(const std::string &name, Callable &&func) {
    supersonic_inlet_[name] = func;
  }
  template <typename Callable>
  void SetSubsonicInlet(const std::string &name, Callable &&func) {
    subsonic_inlet_[name] = func;
  }
  template <typename Callable>
  void SetSubsonicOutlet(const std::string &name, Callable &&func) {
    subsonic_outlet_[name] = func;
  }
  template <typename Callable>
  void SetNoSlipWall(const std::string &name, Callable &&func) {
    no_slip_wall_[name] = func;
  }
  void SetInviscidWall(const std::string &name) {
    inviscid_wall_.emplace_back(name);
  }
  void SetSupersonicOutlet(const std::string &name) {
    supersonic_outlet_.emplace_back(name);
  }

 public:  // implement pure virtual methods declared in Temporal
  void SetTime(double t_curr) override {
    t_curr_ = t_curr;
  }
  void SetSolutionColumn(Column const &column) override {
    for (Cell *cell_ptr: this->part_ptr_->GetLocalCellPointers()) {
      auto i_cell = cell_ptr->id();
      Scalar const *data = AddCellDataOffset(column, i_cell);
      data = cell_ptr->projection().GetCoeffFrom(data);
      assert(data == column.data() + column.size()
          || data == AddCellDataOffset(column, i_cell + 1));
    }
    // this->SetValueNoSlipWalls();
  }
  Column GetSolutionColumn() const override {
    auto column = Column(cell_data_size_);
    for (const Cell &cell : part().GetLocalCells()) {
      auto i_cell = cell.id();
      Scalar *data = AddCellDataOffset(&column, i_cell);
      data = cell.projection().WriteCoeffTo(data);
      assert(data == column.data() + column.size()
          || data == AddCellDataOffset(column, i_cell + 1));
    }
    return column;
  }
  Column GetResidualColumn() const override {
    this->part_ptr_->ShareGhostCellCoeffs();
    auto residual = Column(cell_data_size_);
    residual.setZero();
    this->AddFluxDivergence(&GetFluxMatrix, &residual);
    this->AddFluxOnLocalFaces(&residual);
    this->AddFluxOnBoundaries(&residual);
    this->part_ptr_->UpdateGhostCellCoeffs();
    this->AddFluxOnGhostFaces(&residual);
    return residual;
  }

  void AddFluxOnBoundaries(Column *residual) const {
#ifdef ENABLE_LOGGING
    log() << "Enter " << fullname() << "::AddFluxOnBoundaries\n";
    log() << fullname() << "::AddFluxOnInviscidWalls\n";
    log() << residual->squaredNorm() << "\n";
#endif
    this->AddFluxOnInviscidWalls(residual);
#ifdef ENABLE_LOGGING
    log() << residual->squaredNorm() << "\n";
    log() << fullname() << "::AddFluxOnSupersonicInlets\n";
    log() << residual->squaredNorm() << "\n";
#endif
    this->AddFluxOnNoSlipWalls(residual);
#ifdef ENABLE_LOGGING
    log() << residual->squaredNorm() << "\n";
    log() << fullname() << "::AddFluxOnNoSlipWalls\n";
    log() << residual->squaredNorm() << "\n";
#endif
    this->AddFluxOnSupersonicInlets(residual);
#ifdef ENABLE_LOGGING
    log() << residual->squaredNorm() << "\n";
    log() << fullname() << "::AddFluxOnSupersonicOutlets\n";
    log() << residual->squaredNorm() << "\n";
#endif
    this->AddFluxOnSupersonicOutlets(residual);
#ifdef ENABLE_LOGGING
    log() << residual->squaredNorm() << "\n";
    log() << fullname() << "::AddFluxOnSubsonicInlets\n";
    log() << residual->squaredNorm() << "\n";
#endif
    this->AddFluxOnSubsonicInlets(residual);
#ifdef ENABLE_LOGGING
    log() << residual->squaredNorm() << "\n";
    log() << fullname() << "::AddFluxOnSubsonicOutlets\n";
    log() << residual->squaredNorm() << "\n";
#endif
    this->AddFluxOnSubsonicOutlets(residual);
#ifdef ENABLE_LOGGING
    log() << residual->squaredNorm() << "\n";
    log() << fullname() << "::AddFluxOnSmartBoundaries\n";
    log() << residual->squaredNorm() << "\n";
#endif
    this->AddFluxOnSmartBoundaries(residual);
#ifdef ENABLE_LOGGING
    log() << residual->squaredNorm() << "\n";
    log() << "Leave " << fullname() << "::AddFluxOnBoundaries\n";
#endif
  }

 public:  // declare pure virtual methods to be implemented in subclasses
  using FluxMatrix = typename Riemann::FluxMatrix;
  using CellToFlux = FluxMatrix (*)(const Cell &cell, int q);

  /**
   * @brief Add the flux divergence to the residual column of the given Cell.
   * 
   * @param cell_to_flux the pointer of a function that gets the FluxMatrix on a given Gaussian point of a Cell
   * @param cell the Cell to be processed
   * @param data the residual column of the given Cell
   */
  virtual void AddFluxDivergence(CellToFlux cell_to_flux, Cell const &cell,
      Scalar *data) const = 0;
  /**
   * @brief Add the flux divergence to the residual column of the given Part.
   * 
   * @param cell_to_flux the pointer of a function that gets the FluxMatrix on a given Gaussian point of a Cell
   * @param data the residual column of the given Part
   */
  virtual void AddFluxDivergence(CellToFlux cell_to_flux,
      Column *residual) const {
    for (const Cell &cell : part().GetLocalCells()) {
      auto *data = this->AddCellDataOffset(residual, cell.id());
      this->AddFluxDivergence(cell_to_flux, cell, data);
    }
  }
  virtual void AddFluxOnLocalFaces(Column *residual) const = 0;
  virtual void AddFluxOnGhostFaces(Column *residual) const = 0;
  virtual void SetValueNoSlipWalls() {
  }
  virtual void AddFluxOnNoSlipWalls(Column *residual) const {
  }
  virtual void AddFluxOnInviscidWalls(Column *residual) const = 0;
  virtual void AddFluxOnSupersonicInlets(Column *residual) const = 0;
  virtual void AddFluxOnSupersonicOutlets(Column *residual) const = 0;
  virtual void AddFluxOnSubsonicInlets(Column *residual) const = 0;
  virtual void AddFluxOnSubsonicOutlets(Column *residual) const = 0;
  virtual void AddFluxOnSmartBoundaries(Column *residual) const = 0;

 public:
  static FluxMatrix GetFluxMatrix(const Projection &projection, int q)
      requires(!mini::riemann::Diffusive<Riemann>) {
    return Riemann::GetFluxMatrix(projection.GetValue(q));
  }
  static FluxMatrix GetFluxMatrix(const Projection &projection, int q)
      requires(mini::riemann::ConvectiveDiffusive<Riemann>) {
    const auto &value = projection.GetValue(q);
    FluxMatrix flux_matrix = Riemann::GetFluxMatrix(value);
    const auto &gradient = projection.GetGlobalGradient(q);
    Riemann::MinusViscousFlux(value, gradient, &flux_matrix);
    return flux_matrix;
  }
  static FluxMatrix GetFluxMatrix(const Cell &cell, int q) {
    return GetFluxMatrix(cell.projection(), q);
  }
};

}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_FEM_HPP_
