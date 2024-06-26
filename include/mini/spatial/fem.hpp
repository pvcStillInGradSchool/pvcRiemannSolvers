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
#include <type_traits>
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

template <typename P, typename R>
class FiniteElement : public temporal::System<typename P::Scalar> {
 public:
  using Part = P;
  using Riemann = R;
  using Scalar = typename Part::Scalar;
  using Face = typename Part::Face;
  using Cell = typename Part::Cell;
  using Index = typename Part::Index;
  using Global = typename Cell::Global;
  using Integrator = typename Cell::Integrator;
  using Polynomial = typename Cell::Polynomial;
  using Coeff = typename Polynomial::Coeff;
  using Value = typename Polynomial::Value;
  using Temporal = temporal::System<typename Part::Scalar>;
  using Column = typename Temporal::Column;

  static_assert(mini::riemann::Convective<Riemann>);
  static_assert(std::is_same_v<Scalar, typename Riemann::Scalar>);
  static_assert(std::is_same_v<Scalar, typename Riemann::Scalar>);
  static_assert(Riemann::kComponents == Polynomial::K);
  static_assert(Riemann::kDimensions == Polynomial::D);

 protected:
  std::vector<std::string> supersonic_outlet_, inviscid_wall_;
  using Function = std::function<Value(const Global &, double)>;
  std::unordered_map<std::string, Function> supersonic_inlet_,
      subsonic_inlet_, subsonic_outlet_, smart_boundary_, no_slip_wall_;

  // [i_face][i_gauss]
  std::vector<std::vector<Riemann>> riemann_;

  void SetDistance(Riemann *riemann_ptr, Face const &face)
      requires(!mini::riemann::Diffusive<R>) {
  }
  void SetDistance(Riemann *riemann_ptr, Face const &face)
      requires(mini::riemann::Diffusive<R>) {
    riemann_ptr->SetDistance(riemann_ptr->normal().dot(face.HolderToSharer()));
  }

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
    auto build_riemanns = [this](std::ranges::input_range auto faces) {
      for (Face const &face : faces) {
        assert(face.id() == this->riemann_.size());
        auto const &integrator = face.integrator();
        auto &riemanns = this->riemann_.emplace_back(integrator.CountPoints());
        for (int i = 0, n = riemanns.size(); i < n; ++i) {
          riemanns.at(i).Rotate(integrator.GetNormalFrame(i));
          SetDistance(&riemanns.at(i), face);
        }
      }
    };
    build_riemanns(part_ptr->GetLocalFaces());
    build_riemanns(part_ptr->GetGhostFaces());
    build_riemanns(part_ptr->GetBoundaryFaces());
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
    assert(part_ptr_);
    return part_ptr_;
  }
  Part *part_ptr() const {
    assert(part_ptr_);
    return part_ptr_;
  }
  Part const &part() const {
    return *part_ptr();
  }

  /**
   * @brief Get the FiniteElement::Riemann solvers on a given FiniteElement::Face.
   * 
   * @param face the given FiniteElement::Face
   * @return a vector of FiniteElement::Riemann solvers indexed by quadrature points.
   */
  std::vector<Riemann> const &GetRiemannSolvers(Face const &face) const {
#ifdef NDEBUG
    return riemann_[face.id()];
#else
    return riemann_.at(face.id());
#endif
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
    for (Cell *cell_ptr: part_ptr()->GetLocalCellPointers()) {
      auto i_cell = cell_ptr->id();
      Scalar const *data = AddCellDataOffset(column, i_cell);
      data = cell_ptr->polynomial().GetCoeffFrom(data);
      assert(data == column.data() + column.size()
          || data == AddCellDataOffset(column, i_cell + 1));
    }
  }
  Column GetSolutionColumn() const override {
    auto column = Column(cell_data_size_);
    for (const Cell &cell : part().GetLocalCells()) {
      auto i_cell = cell.id();
      Scalar *data = AddCellDataOffset(&column, i_cell);
      data = cell.polynomial().WriteCoeffTo(data);
      assert(data == column.data() + column.size()
          || data == AddCellDataOffset(column, i_cell + 1));
    }
    return column;
  }
  Column GetResidualColumn() const override {
    part_ptr()->ShareGhostCellCoeffs();
    auto residual = Column(cell_data_size_);
    residual.setZero();
    this->AddFluxDivergenceOnLocalCells(&residual);
    this->AddFluxOnLocalFaces(&residual);
    this->AddFluxOnBoundaries(&residual);
    part_ptr()->UpdateGhostCellCoeffs();
    this->AddFluxOnGhostFaces(&residual);
    return residual;
  }

  void AddFluxOnBoundaries(Column *residual) const {
#ifdef ENABLE_LOGGING
    log() << "Enter " << fullname() << "::AddFluxOnBoundaries\n";
    log() << fullname() << "::AddFluxOnSupersonicInlets\n";
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
    log() << fullname() << "::AddFluxOnInviscidWalls\n";
    log() << residual->squaredNorm() << "\n";
#endif
    this->AddFluxOnInviscidWalls(residual);
#ifdef ENABLE_LOGGING
    log() << residual->squaredNorm() << "\n";
    log() << fullname() << "::AddFluxOnNoSlipWalls\n";
    log() << residual->squaredNorm() << "\n";
#endif
    this->AddFluxOnNoSlipWalls(residual);
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

  /**
   * @brief Add the flux divergence to the residual column of the given FiniteElement::Cell.
   * 
   * @param cell the FiniteElement::Cell to be processed
   * @param residual the residual column of the given FiniteElement::Cell
   */
  virtual void AddFluxDivergence(Cell const &cell, Scalar *residual) const = 0;

  /**
   * @brief Add the flux divergence to the residual FiniteElement::Column of the given FiniteElement::Part.
   * 
   * It delegates the work to the pure virtual method FiniteElement::AddFluxDivergence, which must be implemented in a concrete class.
   * 
   * @param residual the residual FiniteElement::Column of the given FiniteElement::Part
   */
  void AddFluxDivergenceOnLocalCells(Column *residual) const {
    if (Part::kDegrees == 0) {
      return;
    }
    for (const Cell &cell : part().GetLocalCells()) {
      auto *data = this->AddCellDataOffset(residual, cell.id());
      this->AddFluxDivergence(cell, data);
    }
  }

  /**
   * @brief Solve the FiniteElement::Riemann problem on the given FiniteElement::Face, then add the flux to the residual of the given FiniteElement::Cell(s).
   * 
   * @param face the FiniteElement::Face to be processed
   * @param holder_data the residual column of the holder FiniteElement::Cell of the given FiniteElement::Face
   * @param sharer_data the residual column of the sharer FiniteElement::Cell of the given FiniteElement::Face
   */
  virtual void AddFluxToHolderAndSharer(Face const &face,
      Scalar *holder_data, Scalar *sharer_data) const = 0;

  virtual void AddFluxToHolder(Face const &face, Scalar *holder_data) const {
  }
  virtual void AddFluxToSharer(Face const &face, Scalar *sharer_data) const {
  }

  /**
   * @brief Add the fluxes on local (requiring no MPI communication) FiniteElement::Face's to the residual FiniteElement::Column of the given FiniteElement::Part.
   * 
   * It delegates the work to the pure virtual method FiniteElement::AddFluxToHolderAndSharer, which must be implemented in a concrete class.
   * 
   * @param residual the residual FiniteElement::Column of the given FiniteElement::Part
   */
  void AddFluxOnLocalFaces(Column *residual) const {
    for (const Face &face : this->part().GetLocalFaces()) {
      this->AddFluxToHolderAndSharer(face,
          this->AddCellDataOffset(residual, face.holder().id()),
          this->AddCellDataOffset(residual, face.sharer().id()));
    }
  }

  /**
   * @brief Add the fluxes on ghost (requiring MPI communications) FiniteElement::Face's to the residual FiniteElement::Column of the given FiniteElement::Part.
   * 
   * It delegates the work to the pure virtual method FiniteElement::AddFluxToHolderAndSharer, which must be implemented in a concrete class.
   * 
   * @param residual the residual FiniteElement::Column of the given FiniteElement::Part
   */
  void AddFluxOnGhostFaces(Column *residual) const {
    for (const Face &face : this->part().GetGhostFaces()) {
      this->AddFluxToHolderAndSharer(face,
          this->AddCellDataOffset(residual, face.holder().id()),
          nullptr);
    }
  }

  virtual void AddFluxOnNoSlipWalls(Column *residual) const {
  }
  virtual void AddFluxOnInviscidWalls(Column *residual) const = 0;
  virtual void AddFluxOnSupersonicInlets(Column *residual) const = 0;
  virtual void AddFluxOnSupersonicOutlets(Column *residual) const = 0;
  virtual void AddFluxOnSubsonicInlets(Column *residual) const = 0;
  virtual void AddFluxOnSubsonicOutlets(Column *residual) const = 0;
  virtual void AddFluxOnSmartBoundaries(Column *residual) const = 0;

 private:
  static FluxMatrix _GetFluxMatrix(const Cell &cell, int q)
      requires(!mini::riemann::Diffusive<Riemann>) {
    return Riemann::GetFluxMatrix(cell.polynomial().GetValue(q));
  }

  static FluxMatrix _GetFluxMatrix(const Cell &cell, int q)
      requires(mini::riemann::ConvectiveDiffusive<Riemann>) {
    auto [value, gradient] = cell.polynomial().GetGlobalValueGradient(q);
    FluxMatrix flux_matrix = Riemann::Convection::GetFluxMatrix(value);
    const auto &property = Riemann::Diffusion::GetPropertyOnCell(cell.id(), q);
    Riemann::MinusViscousFlux(&flux_matrix, property, value, gradient);
    return flux_matrix;
  }

 public:
  static FluxMatrix GetFluxMatrix(const Cell &cell, int q) {
    return _GetFluxMatrix(cell, q);
  }
};

}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_FEM_HPP_
