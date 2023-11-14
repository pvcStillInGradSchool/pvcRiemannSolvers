// Copyright 2023 PEI Weicheng
#ifndef MINI_SPATIAL_FEM_HPP_
#define MINI_SPATIAL_FEM_HPP_

#include <cassert>
#include <functional>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "mini/temporal/ode.hpp"

namespace mini {
namespace spatial {

template <typename P>
class DummySource {
 public:
  using Part = P;
  using Cell = typename Part::Cell;
  using Coeff = typename Cell::Projection::Coeff;

  void UpdateCoeff(const Cell &cell, double t_curr, Coeff *coeff) {
  }
};

template <typename Part>
class FiniteElement : public temporal::System<typename Part::Scalar> {
 public:
  using Riemann = typename Part::Riemann;
  using Scalar = typename Part::Scalar;
  using Face = typename Part::Face;
  using Cell = typename Part::Cell;
  using Global = typename Cell::Global;
  using Projection = typename Cell::Projection;
  using Coeff = typename Projection::Coeff;
  using Value = typename Projection::Value;
  using Temporal = temporal::System<typename Part::Scalar>;
  using Column = typename Temporal::Column;

 protected:
  std::vector<std::string> supersonic_outlet_, solid_wall_;
  using Function = std::function<Value(const Global &, double)>;
  std::unordered_map<std::string, Function> supersonic_inlet_,
      subsonic_inlet_, subsonic_outlet_, smart_boundary_;

  Part *part_ptr_;
  double t_curr_;
  size_t cell_data_size_;

 public:
  explicit FiniteElement(Part *part_ptr)
      : part_ptr_(part_ptr), cell_data_size_(this->part_ptr_->GetCellDataSize()) {
  }
  FiniteElement(const FiniteElement &) = default;
  FiniteElement &operator=(const FiniteElement &) = default;
  FiniteElement(FiniteElement &&) noexcept = default;
  FiniteElement &operator=(FiniteElement &&) noexcept = default;
  ~FiniteElement() noexcept = default;

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
  void SetSolidWall(const std::string &name) {
    solid_wall_.emplace_back(name);
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
      Scalar const *data = column.data() + this->part_ptr_->GetCellDataOffset(i_cell);
      cell_ptr->projection_.GetCoeffFrom(data);
    }
  }
  Column GetSolutionColumn() const override {
    auto column = Column(cell_data_size_);
    for (const auto &cell : this->part_ptr_->GetLocalCells()) {
      auto i_cell = cell.id();
      Scalar *data = column.data() + this->part_ptr_->GetCellDataOffset(i_cell);
      cell.projection_.WriteCoeffTo(data);
    }
    return column;
  }
  Column GetResidualColumn() const override {
    this->part_ptr_->ShareGhostCellCoeffs();
    auto residual = Column(cell_data_size_);
    residual.setZero();
    this->AddFluxDivergence(&residual);
    this->AddFluxOnLocalFaces(&residual);
    this->AddFluxOnBoundaries(&residual);
    this->part_ptr_->UpdateGhostCellCoeffs();
    this->AddFluxOnGhostFaces(&residual);
    return residual;
  }
  
 protected:  // declare pure virtual methods to be implemented in subclasses
  virtual void AddFluxDivergence(Column *residual) const = 0;
  virtual void AddFluxOnLocalFaces(Column *residual) const = 0;
  virtual void AddFluxOnBoundaries(Column *residual) const = 0;
  virtual void AddFluxOnGhostFaces(Column *residual) const = 0;
};

namespace fem {

template <typename Part>
class DiscontinuousGalerkin : public FiniteElement<Part> {
  using Base = FiniteElement<Part>;

 public:
  using Riemann = typename Base::Riemann;
  using Scalar = typename Base::Scalar;
  using Face = typename Base::Face;
  using Cell = typename Base::Cell;
  using Global = typename Base::Global;
  using Projection = typename Base::Projection;
  using Coeff = typename Base::Coeff;
  using Value = typename Base::Value;
  using Temporal = typename Base::Temporal;
  using Column = typename Base::Column;

 public:
  explicit DiscontinuousGalerkin(Part *part_ptr)
      : Base(part_ptr) {
  }
  DiscontinuousGalerkin(const DiscontinuousGalerkin &) = default;
  DiscontinuousGalerkin &operator=(const DiscontinuousGalerkin &) = default;
  DiscontinuousGalerkin(DiscontinuousGalerkin &&) noexcept = default;
  DiscontinuousGalerkin &operator=(DiscontinuousGalerkin &&) noexcept = default;
  ~DiscontinuousGalerkin() noexcept = default;

 protected:  // implement pure virtual methods declared in Base
  void AddFluxDivergence(Column *residual) const override {
    // Integrate the source term, if there is any.
    // if (!std::is_same_v<Source, DummySource<Part>>) {
    //   for (const Cell &cell : this->part_ptr_->GetLocalCells()) {
    //     auto &coeff = residual_.at(cell.id());
    //     source_.UpdateCoeff(cell, this->t_curr_, &coeff);
    //   }
    // }
    // Integrate the dot-product of flux and basis gradient, if there is any.
    if (Part::kDegrees > 0) {
      for (const Cell &cell : this->part_ptr_->GetLocalCells()) {
        auto i_cell = cell.id();
        auto *data = residual->data() + this->part_ptr_->GetCellDataOffset(i_cell);
        const auto &gauss = cell.gauss();
        for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
          const auto &xyz = gauss.GetGlobalCoord(q);
          Value cv = cell.GlobalToValue(xyz);
          auto flux = Riemann::GetFluxMatrix(cv);
          auto grad = cell.projection_.GlobalToBasisGradients(xyz);
          Coeff prod = flux * grad;
          prod *= gauss.GetGlobalWeight(q);
          cell.projection_.AddCoeffTo(prod, data);
        }
      }
    }
  }
  void AddFluxOnLocalFaces(Column *residual) const override {
    for (const Face &face : this->part_ptr_->GetLocalFaces()) {
      const auto &gauss = face.gauss();
      const auto &holder = face.holder();
      const auto &sharer = face.sharer();
      const auto &riemann = face.riemann();
      auto *holder_data = residual->data()
          + this->part_ptr_->GetCellDataOffset(holder.id());
      auto *sharer_data = residual->data()
          + this->part_ptr_->GetCellDataOffset(sharer.id());
      for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
        const auto &coord = gauss.GetGlobalCoord(q);
        Value u_holder = holder.GlobalToValue(coord);
        Value u_sharer = sharer.GlobalToValue(coord);
        Value flux = riemann.GetFluxUpwind(u_holder, u_sharer);
        flux *= gauss.GetGlobalWeight(q);
        Coeff prod = -flux * holder.GlobalToBasisValues(coord);
        holder.projection_.AddCoeffTo(prod, holder_data);
        prod = flux * sharer.GlobalToBasisValues(coord);
        sharer.projection_.AddCoeffTo(prod, sharer_data);
      }
    }
  }
  void AddFluxOnGhostFaces(Column *residual) const override {
    for (const Face &face : this->part_ptr_->GetGhostFaces()) {
      const auto &gauss = face.gauss();
      const auto &holder = face.holder();
      const auto &sharer = face.sharer();
      const auto &riemann = face.riemann();
      auto *holder_data = residual->data()
          + this->part_ptr_->GetCellDataOffset(holder.id());
      for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
        const auto &coord = gauss.GetGlobalCoord(q);
        Value u_holder = holder.GlobalToValue(coord);
        Value u_sharer = sharer.GlobalToValue(coord);
        Value flux = riemann.GetFluxUpwind(u_holder, u_sharer);
        flux *= gauss.GetGlobalWeight(q);
        Coeff prod = -flux * holder.GlobalToBasisValues(coord);
        holder.projection_.AddCoeffTo(prod, holder_data);
      }
    }
  }
  void AddFluxOnBoundaries(Column *residual) const override {
    this->ApplySolidWall(residual);
    this->ApplySupersonicInlet(residual);
    this->ApplySupersonicOutlet(residual);
    this->ApplySubsonicInlet(residual);
    this->ApplySubsonicOutlet(residual);
    this->ApplySmartBoundary(residual);
  }

 protected:  // virtual methods that might be overriden in subclasses
  virtual void ApplySolidWall(Column *residual) const {
    for (const auto &name : this->solid_wall_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        const auto &riemann = face.riemann();
        auto *holder_data = residual->data()
            + this->part_ptr_->GetCellDataOffset(holder.id());
        for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
          const auto &coord = gauss.GetGlobalCoord(q);
          Value u_holder = holder.GlobalToValue(coord);
          Value flux = riemann.GetFluxOnSolidWall(u_holder);
          flux *= gauss.GetGlobalWeight(q);
          Coeff prod = -flux * holder.GlobalToBasisValues(coord);
          holder.projection_.AddCoeffTo(prod, holder_data);
        }
      }
    }
  }
  virtual void ApplySupersonicOutlet(Column *residual) const {
    for (const auto &name : this->supersonic_outlet_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        const auto &riemann = face.riemann();
        auto *holder_data = residual->data()
            + this->part_ptr_->GetCellDataOffset(holder.id());
        for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
          const auto &coord = gauss.GetGlobalCoord(q);
          Value u_holder = holder.GlobalToValue(coord);
          Value flux = riemann.GetFluxOnSupersonicOutlet(u_holder);
          flux *= gauss.GetGlobalWeight(q);
          Coeff prod = -flux * holder.GlobalToBasisValues(coord);
          holder.projection_.AddCoeffTo(prod, holder_data);
        }
      }
    }
  }
  virtual void ApplySupersonicInlet(Column *residual) const {
    for (auto &[name, func] : this->supersonic_inlet_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        const auto &riemann = face.riemann();
        auto *holder_data = residual->data()
            + this->part_ptr_->GetCellDataOffset(holder.id());
        for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
          const auto &coord = gauss.GetGlobalCoord(q);
          Value u_given = func(coord, this->t_curr_);
          Value flux = riemann.GetFluxOnSupersonicInlet(u_given);
          flux *= gauss.GetGlobalWeight(q);
          Coeff prod = -flux * holder.GlobalToBasisValues(coord);
          holder.projection_.AddCoeffTo(prod, holder_data);
        }
      }
    }
  }
  virtual void ApplySubsonicInlet(Column *residual) const {
    for (auto &[name, func] : this->subsonic_inlet_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        const auto &riemann = face.riemann();
        auto *holder_data = residual->data()
            + this->part_ptr_->GetCellDataOffset(holder.id());
        for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
          const auto &coord = gauss.GetGlobalCoord(q);
          Value u_inner = holder.GlobalToValue(coord);
          Value u_given = func(coord, this->t_curr_);
          Value flux = riemann.GetFluxOnSubsonicInlet(u_inner, u_given);
          flux *= gauss.GetGlobalWeight(q);
          Coeff prod = -flux * holder.GlobalToBasisValues(coord);
          holder.projection_.AddCoeffTo(prod, holder_data);
        }
      }
    }
  }
  virtual void ApplySubsonicOutlet(Column *residual) const {
    for (auto &[name, func] : this->subsonic_outlet_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        const auto &riemann = face.riemann();
        auto *holder_data = residual->data()
            + this->part_ptr_->GetCellDataOffset(holder.id());
        for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
          const auto &coord = gauss.GetGlobalCoord(q);
          Value u_inner = holder.GlobalToValue(coord);
          Value u_given = func(coord, this->t_curr_);
          Value flux = riemann.GetFluxOnSubsonicOutlet(u_inner, u_given);
          flux *= gauss.GetGlobalWeight(q);
          Coeff prod = -flux * holder.GlobalToBasisValues(coord);
          holder.projection_.AddCoeffTo(prod, holder_data);
        }
      }
    }
  }
  virtual void ApplySmartBoundary(Column *residual) const {
    for (auto &[name, func] : this->smart_boundary_) {
      for (const Face &face : this->part_ptr_->GetBoundaryFaces(name)) {
        const auto &gauss = face.gauss();
        const auto &holder = face.holder();
        const auto &riemann = face.riemann();
        auto *holder_data = residual->data()
            + this->part_ptr_->GetCellDataOffset(holder.id());
        for (int q = 0, n = gauss.CountPoints(); q < n; ++q) {
          const auto &coord = gauss.GetGlobalCoord(q);
          Value u_inner = holder.GlobalToValue(coord);
          Value u_given = func(coord, this->t_curr_);
          Value flux = riemann.GetFluxOnSmartBoundary(u_inner, u_given);
          flux *= gauss.GetGlobalWeight(q);
          Coeff prod = -flux * holder.GlobalToBasisValues(coord);
          holder.projection_.AddCoeffTo(prod, holder_data);
        }
      }
    }
  }
};

template <typename Part, typename Limiter, typename Source = DummySource<Part>>
class DGwithLimiterAndSource : public DiscontinuousGalerkin<Part> {
  using Base = DiscontinuousGalerkin<Part>;

 public:
  using Riemann = typename Base::Riemann;
  using Scalar = typename Base::Scalar;
  using Face = typename Base::Face;
  using Cell = typename Base::Cell;
  using Global = typename Base::Global;
  using Projection = typename Base::Projection;
  using Coeff = typename Base::Coeff;
  using Value = typename Base::Value;
  using Temporal = typename Base::Temporal;
  using Column = typename Base::Column;

 protected:
  Limiter limiter_;
  Source source_;

 public:
  DGwithLimiterAndSource(Part *part_ptr,
          const Limiter &limiter, const Source &source = Source())
      : Base(part_ptr), limiter_(limiter), source_(source) {
  }
  DGwithLimiterAndSource(const DGwithLimiterAndSource &) = default;
  DGwithLimiterAndSource &operator=(const DGwithLimiterAndSource &) = default;
  DGwithLimiterAndSource(DGwithLimiterAndSource &&) noexcept = default;
  DGwithLimiterAndSource &operator=(DGwithLimiterAndSource &&) noexcept = default;
  ~DGwithLimiterAndSource() noexcept = default;

 public:  // implement pure virtual methods declared in Temporal
  void SetSolutionColumn(Column const &column) override {
    this->Base::SetSolutionColumn(column);
    this->part_ptr_->Reconstruct(limiter_);
  }
};

}  // namespace fem
}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_FEM_HPP_
