// Copyright 2023 PEI Weicheng
#ifndef MINI_SPATIAL_DG_GENERAL_HPP_
#define MINI_SPATIAL_DG_GENERAL_HPP_

#include <cassert>
#include <functional>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "mini/spatial/fem.hpp"
#include "mini/limiter/reconstruct.hpp"

namespace mini {
namespace spatial {
namespace dg {

template <typename P, typename R>
class General : public spatial::FiniteElement<P, R> {
 public:
  using Base = spatial::FiniteElement<P, R>;
  using Part = typename Base::Part;
  using Riemann = typename Base::Riemann;
  using Scalar = typename Base::Scalar;
  using Face = typename Base::Face;
  using Cell = typename Base::Cell;
  using Global = typename Base::Global;
  using Polynomial = typename Base::Polynomial;
  using Coeff = typename Base::Coeff;
  using Value = typename Base::Value;
  using Temporal = typename Base::Temporal;
  using Column = typename Base::Column;

 public:
  explicit General(Part *part_ptr)
      : Base(part_ptr) {
  }
  General(const General &) = default;
  General &operator=(const General &) = default;
  General(General &&) noexcept = default;
  General &operator=(General &&) noexcept = default;
  ~General() noexcept = default;

  std::string name() const override {
    return "DG::General";
  }

  virtual Value GetValueJump(Face const &face, int i_flux_point) const {
    const auto &global = face.integrator().GetGlobal(i_flux_point);
    return face.holder().polynomial().GlobalToValue(global)
         - face.sharer().polynomial().GlobalToValue(global);
  }

 protected:  // implement pure virtual methods declared in Base
  void AddFluxDivergence(Cell const &cell, Scalar *residual) const override {
    assert(residual);
    const auto &integrator = cell.integrator();
    for (int q = 0, n = integrator.CountPoints(); q < n; ++q) {
      const auto &xyz = integrator.GetGlobal(q);
      auto flux = Base::GetFluxMatrix(cell, q);
      flux *= integrator.GetGlobalWeight(q);
      auto grad = cell.polynomial().GlobalToBasisGlobalGradients(xyz);
      Coeff prod = flux * grad;
      Polynomial::AddToResidual(prod, residual);
    }
  }
  void AddFluxToHolderAndSharer(Face const &face,
      Scalar *holder_data, Scalar *sharer_data) const override {
    const auto &riemanns = this->GetRiemannSolvers(face);
    const auto &integrator = face.integrator();
    const auto &holder = face.holder();
    const auto &sharer = face.sharer();
    for (int q = 0, n = integrator.CountPoints(); q < n; ++q) {
      const auto &coord = integrator.GetGlobal(q);
      Value u_holder = holder.GlobalToValue(coord);
      Value u_sharer = sharer.GlobalToValue(coord);
      Value flux = riemanns[q].GetFluxUpwind(u_holder, u_sharer);
      flux *= -integrator.GetGlobalWeight(q);
      Coeff prod = flux * holder.GlobalToBasisValues(coord);
      assert(holder_data);
      Polynomial::AddToResidual(prod, holder_data);
      if (nullptr == sharer_data) { continue; }
      prod = -flux * sharer.GlobalToBasisValues(coord);
      Polynomial::AddToResidual(prod, sharer_data);
    }
  }

 protected:  // virtual methods that might be overriden in subclasses
  void AddFluxOnInviscidWalls(Column *residual) const override {
    for (const auto &name : this->inviscid_wall_) {
      for (const Face &face : this->part().GetBoundaryFaces(name)) {
        const auto &riemanns = this->GetRiemannSolvers(face);
        const auto &integrator = face.integrator();
        const auto &holder = face.holder();
        Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
        for (int q = 0, n = integrator.CountPoints(); q < n; ++q) {
          const auto &coord = integrator.GetGlobal(q);
          Value u_holder = holder.GlobalToValue(coord);
          Value flux = riemanns[q].GetFluxOnInviscidWall(u_holder);
          flux *= -integrator.GetGlobalWeight(q);
          Coeff prod = flux * holder.GlobalToBasisValues(coord);
          Polynomial::AddToResidual(prod, holder_data);
        }
      }
    }
  }
  void AddFluxOnSupersonicOutlets(Column *residual) const override {
    for (const auto &name : this->supersonic_outlet_) {
      for (const Face &face : this->part().GetBoundaryFaces(name)) {
        const auto &riemanns = this->GetRiemannSolvers(face);
        const auto &integrator = face.integrator();
        const auto &holder = face.holder();
        Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
        for (int q = 0, n = integrator.CountPoints(); q < n; ++q) {
          const auto &coord = integrator.GetGlobal(q);
          Value u_holder = holder.GlobalToValue(coord);
          Value flux = riemanns[q].GetFluxOnSupersonicOutlet(u_holder);
          flux *= -integrator.GetGlobalWeight(q);
          Coeff prod = flux * holder.GlobalToBasisValues(coord);
          Polynomial::AddToResidual(prod, holder_data);
        }
      }
    }
  }
  void AddFluxOnSupersonicInlets(Column *residual) const override {
    for (auto &[name, func] : this->supersonic_inlet_) {
      for (const Face &face : this->part().GetBoundaryFaces(name)) {
        const auto &riemanns = this->GetRiemannSolvers(face);
        const auto &integrator = face.integrator();
        const auto &holder = face.holder();
        Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
        for (int q = 0, n = integrator.CountPoints(); q < n; ++q) {
          const auto &coord = integrator.GetGlobal(q);
          Value u_given = func(coord, this->t_curr_);
          Value flux = riemanns[q].GetFluxOnSupersonicInlet(u_given);
          flux *= -integrator.GetGlobalWeight(q);
          Coeff prod = flux * holder.GlobalToBasisValues(coord);
          Polynomial::AddToResidual(prod, holder_data);
        }
      }
    }
  }
  void AddFluxOnSubsonicInlets(Column *residual) const override {
    for (auto &[name, func] : this->subsonic_inlet_) {
      for (const Face &face : this->part().GetBoundaryFaces(name)) {
        const auto &riemanns = this->GetRiemannSolvers(face);
        const auto &integrator = face.integrator();
        const auto &holder = face.holder();
        Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
        for (int q = 0, n = integrator.CountPoints(); q < n; ++q) {
          const auto &coord = integrator.GetGlobal(q);
          Value u_inner = holder.GlobalToValue(coord);
          Value u_given = func(coord, this->t_curr_);
          Value flux = riemanns[q].GetFluxOnSubsonicInlet(u_inner, u_given);
          flux *= -integrator.GetGlobalWeight(q);
          Coeff prod = flux * holder.GlobalToBasisValues(coord);
          Polynomial::AddToResidual(prod, holder_data);
        }
      }
    }
  }
  void AddFluxOnSubsonicOutlets(Column *residual) const override {
    for (auto &[name, func] : this->subsonic_outlet_) {
      for (const Face &face : this->part().GetBoundaryFaces(name)) {
        const auto &riemanns = this->GetRiemannSolvers(face);
        const auto &integrator = face.integrator();
        const auto &holder = face.holder();
        Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
        for (int q = 0, n = integrator.CountPoints(); q < n; ++q) {
          const auto &coord = integrator.GetGlobal(q);
          Value u_inner = holder.GlobalToValue(coord);
          Value u_given = func(coord, this->t_curr_);
          Value flux = riemanns[q].GetFluxOnSubsonicOutlet(u_inner, u_given);
          flux *= -integrator.GetGlobalWeight(q);
          Coeff prod = flux * holder.GlobalToBasisValues(coord);
          Polynomial::AddToResidual(prod, holder_data);
        }
      }
    }
  }
  void AddFluxOnSmartBoundaries(Column *residual) const override {
    for (auto &[name, func] : this->smart_boundary_) {
      for (const Face &face : this->part().GetBoundaryFaces(name)) {
        const auto &riemanns = this->GetRiemannSolvers(face);
        const auto &integrator = face.integrator();
        const auto &holder = face.holder();
        Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
        for (int q = 0, n = integrator.CountPoints(); q < n; ++q) {
          const auto &coord = integrator.GetGlobal(q);
          Value u_inner = holder.GlobalToValue(coord);
          Value u_given = func(coord, this->t_curr_);
          Value flux = riemanns[q].GetFluxOnSmartBoundary(u_inner, u_given);
          flux *= -integrator.GetGlobalWeight(q);
          Coeff prod = flux * holder.GlobalToBasisValues(coord);
          Polynomial::AddToResidual(prod, holder_data);
        }
      }
    }
  }
};

template <typename P, typename R, typename S>
class WithSource : public General<P, R> {
 public:
  using Source = S;
  using Base = General<P, R>;
  using Part = typename Base::Part;
  using Riemann = typename Base::Riemann;
  using Scalar = typename Base::Scalar;
  using Face = typename Base::Face;
  using Cell = typename Base::Cell;
  using Global = typename Base::Global;
  using Polynomial = typename Base::Polynomial;
  using Coeff = typename Base::Coeff;
  using Value = typename Base::Value;
  using Temporal = typename Base::Temporal;
  using Column = typename Base::Column;

 protected:
  Source *source_ptr_;

 public:
  template <class... Args>
  WithSource(Source *source_ptr, Args&&... args)
      : Base(std::forward<Args>(args)...), source_ptr_(source_ptr) {
  }
  WithSource(const WithSource &) = default;
  WithSource &operator=(const WithSource &) = default;
  WithSource(WithSource &&) noexcept = default;
  WithSource &operator=(WithSource &&) noexcept = default;
  ~WithSource() noexcept = default;

 public:  // implement pure virtual methods declared in Temporal
  Column GetResidualColumn() const override {
    auto residual = this->Base::GetResidualColumn();
    this->AddSourceIntegral(&residual);
    return residual;
  }

 protected:
  virtual void AddSourceIntegral(Column *residual) const {
    // Integrate the source term, if there is any.
    for (const Cell &cell : this->part().GetLocalCells()) {
      Scalar *data = this->AddCellDataOffset(residual, cell.id());
      source_ptr_->UpdateCoeff(cell, this->t_curr_, data);
    }
  }
};

}  // namespace dg
}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_DG_GENERAL_HPP_
