// Copyright 2023 PEI Weicheng
#ifndef MINI_SPATIAL_DG_LOBATTO_HPP_
#define MINI_SPATIAL_DG_LOBATTO_HPP_

#include <concepts>
#include <ranges>

#include <array>
#include <cassert>
#include <functional>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "mini/spatial/dg/general.hpp"
#include "mini/polynomial/hexahedron.hpp"

namespace mini {
namespace spatial {
namespace dg {

/**
 * @brief A specialized version of DG using a Lagrange expansion on Lobatto roots. 
 * 
 * @tparam Part 
 */
template <typename Part>
class Lobatto : public General<Part> {
 public:
  using Base = General<Part>;
  using Riemann = typename Base::Riemann;
  using Scalar = typename Base::Scalar;
  using Face = typename Base::Face;
  using Cell = typename Base::Cell;
  using Global = typename Base::Global;
  using Integrator = typename Base::Integrator;
  using Projection = typename Base::Projection;
  using Coeff = typename Base::Coeff;
  using Value = typename Base::Value;
  using Temporal = typename Base::Temporal;
  using Column = typename Base::Column;

 protected:
  static constexpr int kLineQ = polynomial::LineIntegrator<Projection>::Q;
  static constexpr int kFaceQ = kLineQ * kLineQ;

  using FaceCache = std::array<int16_t, kFaceQ>;
  std::vector<FaceCache> i_node_on_holder_;
  std::vector<FaceCache> i_node_on_sharer_;

  template <std::ranges::input_range R, class FaceToCell>
  void MatchIntegratorianPoints(R &&faces, FaceToCell &&face_to_cell,
      std::vector<FaceCache> *cache) {
    for (const Face &face : faces) {
      assert(cache->size() == face.id());
      auto &curr_face = cache->emplace_back();
      const auto &face_integrator = face.integrator();
      const auto &cell_integrator = face_to_cell(face).integrator();
      for (int f = 0, F = face_integrator.CountPoints(); f < F; ++f) {
        auto &flux_point = face_integrator.GetGlobal(f);
        curr_face.at(f) = -1;
        for (int h = 0, H = cell_integrator.CountPoints(); h < H; ++h) {
          if (Near(flux_point, cell_integrator.GetGlobal(h))) {
            curr_face[f] = h;
            break;
          }
        }
        assert(curr_face[f] >= 0);
      }
    }
  }

  static constexpr bool kLocal = Projection::kLocal;
  static Scalar GetWeight(const Integrator &integrator, int q) requires(kLocal) {
    return integrator.GetLocalWeight(q);
  }
  static Scalar GetWeight(const Integrator &integrator, int q) requires(!kLocal) {
    return integrator.GetGlobalWeight(q);
  }
  using FluxMatrix = typename Riemann::FluxMatrix;
  using CellToFlux = typename Base::CellToFlux;
  static FluxMatrix GetWeightedFluxMatrix(CellToFlux cell_to_flux,
      const Cell &cell, int q) requires(kLocal) {
    auto flux = cell_to_flux(cell, q);
    flux = cell.projection().GlobalFluxToLocalFlux(flux, q);
    flux *= GetWeight(cell.integrator(), q);
    return flux;
  }
  static FluxMatrix GetWeightedFluxMatrix(CellToFlux cell_to_flux,
      const Cell &cell, int q) requires(!kLocal) {
    auto flux = cell_to_flux(cell, q);
    flux *= GetWeight(cell.integrator(), q);
    return flux;
  }

 public:
  explicit Lobatto(Part *part_ptr)
      : Base(part_ptr) {
    auto face_to_holder = [](auto &face) -> auto & { return face.holder(); };
    auto face_to_sharer = [](auto &face) -> auto & { return face.sharer(); };
    auto local_cells = this->part().GetLocalFaces();
    MatchIntegratorianPoints(local_cells, face_to_holder, &i_node_on_holder_);
    MatchIntegratorianPoints(local_cells, face_to_sharer, &i_node_on_sharer_);
    auto ghost_cells = this->part().GetGhostFaces();
    MatchIntegratorianPoints(ghost_cells, face_to_holder, &i_node_on_holder_);
    MatchIntegratorianPoints(ghost_cells, face_to_sharer, &i_node_on_sharer_);
    auto boundary_cells = this->part().GetBoundaryFaces();
    MatchIntegratorianPoints(boundary_cells, face_to_holder, &i_node_on_holder_);
  }
  Lobatto(const Lobatto &) = default;
  Lobatto &operator=(const Lobatto &) = default;
  Lobatto(Lobatto &&) noexcept = default;
  Lobatto &operator=(Lobatto &&) noexcept = default;
  ~Lobatto() noexcept = default;

  std::string name() const override {
    return "DG::Lobatto";
  }

  Column GetResidualColumn() const override {
    Column residual = this->Base::GetResidualColumn();
    // divide mass matrix for each cell
    for (const Cell &cell : this->part().GetLocalCells()) {
      auto i_cell = cell.id();
      Scalar *data = this->AddCellDataOffset(&residual, i_cell);
      const auto &integrator = cell.integrator();
      for (int q = 0, n = integrator.CountPoints(); q < n; ++q) {
        auto scale = 1.0 / GetWeight(integrator, q);
        data = cell.projection().ScaleValueAt(scale, data);
      }
      assert(data == residual.data() + residual.size()
          || data == this->AddCellDataOffset(&residual, i_cell + 1));
    }
    return residual;
  }

 protected:  // override virtual methods defined in Base
  void AddFluxDivergence(CellToFlux cell_to_flux, Cell const &cell,
      Scalar *data) const override {
    const auto &integrator = cell.integrator();
    for (int q = 0, n = integrator.CountPoints(); q < n; ++q) {
      auto flux = GetWeightedFluxMatrix(cell_to_flux, cell, q);
      auto const &grad = cell.projection().GetBasisGradients(q);
      Coeff prod = flux * grad;
      cell.projection().AddCoeffTo(prod, data);
    }
  }
  void AddFluxDivergence(CellToFlux cell_to_flux,
      Column *residual) const override {
    return this->Base::AddFluxDivergence(cell_to_flux, residual);
  }
  void AddFluxOnLocalFaces(Column *residual) const override {
    for (const Face &face : this->part().GetLocalFaces()) {
      const auto &integrator = face.integrator();
      const auto &holder = face.holder();
      const auto &sharer = face.sharer();
      Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
      Scalar *sharer_data = this->AddCellDataOffset(residual, sharer.id());
      auto &i_node_on_holder = i_node_on_holder_[face.id()];
      auto &i_node_on_sharer = i_node_on_sharer_[face.id()];
      for (int f = 0, n = integrator.CountPoints(); f < n; ++f) {
        auto c_holder = i_node_on_holder[f];
        auto c_sharer = i_node_on_sharer[f];
        Value u_holder = holder.projection().GetValue(c_holder);
        Value u_sharer = sharer.projection().GetValue(c_sharer);
        Value flux = face.riemann(f).GetFluxUpwind(u_holder, u_sharer);
        flux *= integrator.GetGlobalWeight(f);
        holder.projection().MinusValue(flux, holder_data, c_holder);
        sharer.projection().AddValueTo(flux, sharer_data, c_sharer);
      }
    }
  }
  void AddFluxOnGhostFaces(Column *residual) const override {
    for (const Face &face : this->part().GetGhostFaces()) {
      const auto &integrator = face.integrator();
      const auto &holder = face.holder();
      const auto &sharer = face.sharer();
      Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
      auto &i_node_on_holder = i_node_on_holder_[face.id()];
      auto &i_node_on_sharer = i_node_on_sharer_[face.id()];
      for (int f = 0, n = integrator.CountPoints(); f < n; ++f) {
        auto c_holder = i_node_on_holder[f];
        auto c_sharer = i_node_on_sharer[f];
        Value u_holder = holder.projection().GetValue(c_holder);
        Value u_sharer = sharer.projection().GetValue(c_sharer);
        Value flux = face.riemann(f).GetFluxUpwind(u_holder, u_sharer);
        flux *= integrator.GetGlobalWeight(f);
        holder.projection().MinusValue(flux, holder_data, c_holder);
      }
    }
  }
  void AddFluxOnInviscidWalls(Column *residual) const override {
    for (const auto &name : this->inviscid_wall_) {
      for (const Face &face : this->part().GetBoundaryFaces(name)) {
        const auto &integrator = face.integrator();
        const auto &holder = face.holder();
        Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
        auto &i_node_on_holder = i_node_on_holder_[face.id()];
        for (int f = 0, n = integrator.CountPoints(); f < n; ++f) {
          auto c_holder = i_node_on_holder[f];
          Value u_holder = holder.projection().GetValue(c_holder);
          Value flux = face.riemann(f).GetFluxOnInviscidWall(u_holder);
          flux *= integrator.GetGlobalWeight(f);
          holder.projection().MinusValue(flux, holder_data, c_holder);
        }
      }
    }
  }
  void AddFluxOnSupersonicOutlets(Column *residual) const override {
    for (const auto &name : this->supersonic_outlet_) {
      for (const Face &face : this->part().GetBoundaryFaces(name)) {
        const auto &integrator = face.integrator();
        const auto &holder = face.holder();
        Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
        auto &i_node_on_holder = i_node_on_holder_[face.id()];
        for (int f = 0, n = integrator.CountPoints(); f < n; ++f) {
          auto c_holder = i_node_on_holder[f];
          Value u_holder = holder.projection().GetValue(c_holder);
          Value flux = face.riemann(f).GetFluxOnSupersonicOutlet(u_holder);
          flux *= integrator.GetGlobalWeight(f);
          holder.projection().MinusValue(flux, holder_data, c_holder);
        }
      }
    }
  }
  void AddFluxOnSupersonicInlets(Column *residual) const override {
    for (auto &[name, func] : this->supersonic_inlet_) {
      for (const Face &face : this->part().GetBoundaryFaces(name)) {
        const auto &integrator = face.integrator();
        const auto &holder = face.holder();
        Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
        auto &i_node_on_holder = i_node_on_holder_[face.id()];
        for (int f = 0, n = integrator.CountPoints(); f < n; ++f) {
          auto c_holder = i_node_on_holder[f];
          Value u_given = func(integrator.GetGlobal(f), this->t_curr_);
          Value flux = face.riemann(f).GetFluxOnSupersonicInlet(u_given);
          flux *= integrator.GetGlobalWeight(f);
          holder.projection().MinusValue(flux, holder_data, c_holder);
        }
      }
    }
  }
  void AddFluxOnSubsonicInlets(Column *residual) const override {
    for (auto &[name, func] : this->subsonic_inlet_) {
      for (const Face &face : this->part().GetBoundaryFaces(name)) {
        const auto &integrator = face.integrator();
        const auto &holder = face.holder();
        Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
        auto &i_node_on_holder = i_node_on_holder_[face.id()];
        for (int f = 0, n = integrator.CountPoints(); f < n; ++f) {
          auto c_holder = i_node_on_holder[f];
          Value u_inner = holder.projection().GetValue(c_holder);
          Value u_given = func(integrator.GetGlobal(f), this->t_curr_);
          Value flux = face.riemann(f).GetFluxOnSubsonicInlet(u_inner, u_given);
          flux *= integrator.GetGlobalWeight(f);
          holder.projection().MinusValue(flux, holder_data, c_holder);
        }
      }
    }
  }
  void AddFluxOnSubsonicOutlets(Column *residual) const override {
    for (auto &[name, func] : this->subsonic_outlet_) {
      for (const Face &face : this->part().GetBoundaryFaces(name)) {
        const auto &integrator = face.integrator();
        const auto &holder = face.holder();
        Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
        auto &i_node_on_holder = i_node_on_holder_[face.id()];
        for (int f = 0, n = integrator.CountPoints(); f < n; ++f) {
          auto c_holder = i_node_on_holder[f];
          Value u_inner = holder.projection().GetValue(c_holder);
          Value u_given = func(integrator.GetGlobal(f), this->t_curr_);
          Value flux = face.riemann(f).GetFluxOnSubsonicOutlet(u_inner, u_given);
          flux *= integrator.GetGlobalWeight(f);
          holder.projection().MinusValue(flux, holder_data, c_holder);
        }
      }
    }
  }
  void AddFluxOnSmartBoundaries(Column *residual) const override {
    for (auto &[name, func] : this->smart_boundary_) {
      for (const Face &face : this->part().GetBoundaryFaces(name)) {
        const auto &integrator = face.integrator();
        const auto &holder = face.holder();
        Scalar *holder_data = this->AddCellDataOffset(residual, holder.id());
        auto &i_node_on_holder = i_node_on_holder_[face.id()];
        for (int f = 0, n = integrator.CountPoints(); f < n; ++f) {
          auto c_holder = i_node_on_holder[f];
          Value u_inner = holder.projection().GetValue(c_holder);
          Value u_given = func(integrator.GetGlobal(f), this->t_curr_);
          Value flux = face.riemann(f).GetFluxOnSmartBoundary(u_inner, u_given);
          flux *= integrator.GetGlobalWeight(f);
          holder.projection().MinusValue(flux, holder_data, c_holder);
        }
      }
    }
  }
};

}  // namespace dg
}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_DG_LOBATTO_HPP_
