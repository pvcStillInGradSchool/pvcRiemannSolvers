// Copyright 2023 PEI Weicheng
#ifndef MINI_SPATIAL_FR_GENERAL_HPP_
#define MINI_SPATIAL_FR_GENERAL_HPP_

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
#include <utility>

#include "mini/polynomial/concept.hpp"
#include "mini/polynomial/hexahedron.hpp"
#include "mini/spatial/fem.hpp"
#include "mini/basis/vincent.hpp"

namespace mini {
namespace spatial {
namespace fr {

/**
 * @brief A general version of FR using a Lagrange expansion whose flux points are also solution points.
 * 
 * @tparam Part 
 * @tparam Riem 
 */
template <typename P, typename R>
    requires mini::polynomial::Nodal<typename P::Polynomial>
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
  static_assert(Polynomial::kLocal);
  using Coeff = typename Base::Coeff;
  using Value = typename Base::Value;
  using Temporal = typename Base::Temporal;
  using Column = typename Base::Column;

 protected:
  using FluxMatrix = typename Riemann::FluxMatrix;

  static constexpr int kLineQ = polynomial::LineIntegrator<Polynomial>::Q;
  static constexpr int kFaceQ = kLineQ * kLineQ;
  static constexpr int kCellQ = kLineQ * kFaceQ;

  struct SolutionPointCache {
    Scalar g_prime;
    int ijk;
  };
  struct FluxPointCache {
    Global normal;  // normal_flux = flux_matrix * normal
    Scalar scale;  // riemann_flux_local = scale * riemann_flux_global
    int ijk;
  };
  using LineCache = std::pair<
      std::array<SolutionPointCache, kLineQ>,
      FluxPointCache
  >;

  using FaceCache = std::array<LineCache, kFaceQ>;
  std::vector<FaceCache> holder_cache_;
  std::vector<FaceCache> sharer_cache_;

  using Vincent = mini::basis::Vincent<Scalar>;
  Vincent vincent_;

  static bool Collinear(Global const &a, Global const &b) {
    return std::abs(1 - std::abs(a.dot(b) / a.norm() / b.norm())) < 1e-8;
  }

  template <std::ranges::input_range Range, class FaceToCell>
  void CacheCorrectionGradients(Range &&faces, FaceToCell &&face_to_cell,
      std::vector<FaceCache> *cache) {
    for (const Face &face : faces) {
      assert(cache->size() == face.id());
      auto &curr_face = cache->emplace_back();
      const auto &face_integrator = face.integrator();
      const auto &cell = face_to_cell(face);
      const auto &cell_integrator = cell.integrator();
      const auto &cell_polynomial = cell.polynomial();
      const auto &cell_basis = cell_polynomial.interpolation().basis();
      int i_face = cell_polynomial.FindFaceId(face.coordinate().center());
      assert(kFaceQ == face.integrator().CountPoints());
      for (int f = 0; f < kFaceQ; ++f) {
        Global const &face_normal = face_integrator.GetNormalFrame(f)[0];
        auto &[curr_line, flux_point] = curr_face.at(f);
        auto &flux_point_coord = face_integrator.GetGlobal(f);
        auto [i, j, k] = cell_polynomial.FindCollinearIndex(flux_point_coord, i_face);
        switch (i_face) {
        case 0:
          assert(k == -1);
          flux_point.ijk = cell_basis.index(i, j, 0);
          assert(Near(flux_point_coord, cell_integrator.GetGlobal(flux_point.ijk)));
          flux_point.normal =
              cell_polynomial.GetJacobianAssociated(flux_point.ijk).col(Z);
          flux_point.scale = -flux_point.normal.norm();
          assert(Collinear(face_normal, flux_point.normal));
          for (k = 0; k < kLineQ; ++k) {
            auto ijk = cell_basis.index(i, j, k);
            auto &local = cell_integrator.GetLocal(ijk);
            auto g_prime = vincent_.LocalToLeftDerivative(local[Z]);
            curr_line[k].g_prime = g_prime;
            curr_line[k].ijk = ijk;
          }
          break;
        case 1:
          assert(j == -1);
          flux_point.ijk = cell_basis.index(i, 0, k);
          assert(Near(flux_point_coord, cell_integrator.GetGlobal(flux_point.ijk)));
          flux_point.normal =
              cell_polynomial.GetJacobianAssociated(flux_point.ijk).col(Y);
          flux_point.scale = -flux_point.normal.norm();
          assert(Collinear(face_normal, flux_point.normal));
          for (j = 0; j < kLineQ; ++j) {
            auto ijk = cell_basis.index(i, j, k);
            auto &local = cell_integrator.GetLocal(ijk);
            auto g_prime = vincent_.LocalToLeftDerivative(local[Y]);
            curr_line[j].g_prime = g_prime;
            curr_line[j].ijk = ijk;
          }
          break;
        case 2:
          assert(i == -1);
          flux_point.ijk = cell_basis.index(kLineQ - 1, j, k);
          assert(Near(flux_point_coord, cell_integrator.GetGlobal(flux_point.ijk)));
          flux_point.normal =
              cell_polynomial.GetJacobianAssociated(flux_point.ijk).col(X);
          flux_point.scale = +flux_point.normal.norm();
          assert(Collinear(face_normal, flux_point.normal));
          for (i = 0; i < kLineQ; ++i) {
            auto ijk = cell_basis.index(i, j, k);
            auto &local = cell_integrator.GetLocal(ijk);
            auto g_prime = vincent_.LocalToRightDerivative(local[X]);
            curr_line[i].g_prime = g_prime;
            curr_line[i].ijk = ijk;
          }
          break;
        case 3:
          assert(j == -1);
          flux_point.ijk = cell_basis.index(i, kLineQ - 1, k);
          assert(Near(flux_point_coord, cell_integrator.GetGlobal(flux_point.ijk)));
          flux_point.normal =
              cell_polynomial.GetJacobianAssociated(flux_point.ijk).col(Y);
          flux_point.scale = +flux_point.normal.norm();
          assert(Collinear(face_normal, flux_point.normal));
          for (j = 0; j < kLineQ; ++j) {
            auto ijk = cell_basis.index(i, j, k);
            auto &local = cell_integrator.GetLocal(ijk);
            auto g_prime = vincent_.LocalToRightDerivative(local[Y]);
            curr_line[j].g_prime = g_prime;
            curr_line[j].ijk = ijk;
          }
          break;
        case 4:
          assert(i == -1);
          flux_point.ijk = cell_basis.index(0, j, k);
          assert(Near(flux_point_coord, cell_integrator.GetGlobal(flux_point.ijk)));
          flux_point.normal =
              cell_polynomial.GetJacobianAssociated(flux_point.ijk).col(X);
          flux_point.scale = -flux_point.normal.norm();
          assert(Collinear(face_normal, flux_point.normal));
          for (i = 0; i < kLineQ; ++i) {
            auto ijk = cell_basis.index(i, j, k);
            auto &local = cell_integrator.GetLocal(ijk);
            auto g_prime = vincent_.LocalToLeftDerivative(local[X]);
            curr_line[i].g_prime = g_prime;
            curr_line[i].ijk = ijk;
          }
          break;
        case 5:
          assert(k == -1);
          flux_point.ijk = cell_basis.index(i, j, kLineQ - 1);
          assert(Near(flux_point_coord, cell_integrator.GetGlobal(flux_point.ijk)));
          flux_point.normal =
              cell_polynomial.GetJacobianAssociated(flux_point.ijk).col(Z);
          flux_point.scale = +flux_point.normal.norm();
          assert(Collinear(face_normal, flux_point.normal));
          for (k = 0; k < kLineQ; ++k) {
            auto ijk = cell_basis.index(i, j, k);
            auto &local = cell_integrator.GetLocal(ijk);
            auto g_prime = vincent_.LocalToRightDerivative(local[Z]);
            curr_line[k].g_prime = g_prime;
            curr_line[k].ijk = ijk;
          }
          break;
        default:
          assert(false);
        }
      }
    }
  }

  // cache of the flux matrices computed in AddFluxDivergence(),
  // reused in GetFluxOnLocalFace() etc.
  std::vector<std::array<FluxMatrix, kCellQ>> flux_matrices_;

 public:
  General(Part *part_ptr, Scalar c_next)
      : Base(part_ptr), vincent_(Part::kDegrees, c_next) {
    auto face_to_holder = [](auto &face) -> auto & { return face.holder(); };
    auto face_to_sharer = [](auto &face) -> auto & { return face.sharer(); };
    auto local_faces = this->part().GetLocalFaces();
    CacheCorrectionGradients(local_faces, face_to_holder, &holder_cache_);
    CacheCorrectionGradients(local_faces, face_to_sharer, &sharer_cache_);
    auto ghost_faces = this->part().GetGhostFaces();
    CacheCorrectionGradients(ghost_faces, face_to_holder, &holder_cache_);
    CacheCorrectionGradients(ghost_faces, face_to_sharer, &sharer_cache_);
    auto boundary_faces = this->part().GetBoundaryFaces();
    CacheCorrectionGradients(boundary_faces, face_to_holder, &holder_cache_);
    flux_matrices_.resize(this->part().CountLocalCells());
  }
  General(const General &) = default;
  General &operator=(const General &) = default;
  General(General &&) noexcept = default;
  General &operator=(General &&) noexcept = default;
  ~General() noexcept = default;

  std::string name() const override {
    return "FR::General";
  }

 protected:  // override virtual methods defined in Base
  using CellToFlux = typename Base::CellToFlux;
  void AddFluxDivergence(CellToFlux cell_to_flux, Cell const &cell,
      Scalar *data) const override {
    const auto &integrator = cell.integrator();
    const auto &polynomial = cell.polynomial();
    auto &flux = const_cast<General *>(this)->flux_matrices_.at(cell.id());
    for (int q = 0, n = integrator.CountPoints(); q < n; ++q) {
      FluxMatrix global_flux = cell_to_flux(cell, q);
      flux[q] = polynomial.GlobalFluxToLocalFlux(global_flux, q);
    }
    for (int q = 0, n = integrator.CountPoints(); q < n; ++q) {
      auto const &grad = polynomial.GetBasisGradients(q);
      Value value = flux[0] * grad.col(0);
      for (int k = 1; k < n; ++k) {
        value += flux[k] * grad.col(k);
      }
      Polynomial::MinusValue(value, data, q);
    }
  }
  static std::pair<Value, Value> GetFluxOnLocalFace(const Riemann &riemann,
      const Polynomial &holder_polynomial, FluxPointCache const &holder_cache,
      const Polynomial &sharer_polynomial, FluxPointCache const &sharer_cache)
      requires(!mini::riemann::Diffusive<Riemann>) {
    Value u_holder = holder_polynomial.GetValue(holder_cache.ijk);
    Value u_sharer = sharer_polynomial.GetValue(sharer_cache.ijk);
    Value f_upwind = riemann.GetFluxUpwind(u_holder, u_sharer);
    assert(Collinear(holder_cache.normal, sharer_cache.normal));
    Value f_holder = f_upwind * holder_cache.scale;
    f_holder -= Riemann::GetFluxMatrix(u_holder) * holder_cache.normal;
    Value f_sharer = f_upwind * (-sharer_cache.scale);
    f_sharer -= Riemann::GetFluxMatrix(u_sharer) * sharer_cache.normal;
    return { f_holder, f_sharer };
  }
  static std::pair<Value, Value> GetFluxOnLocalFace(const Riemann &riemann,
      const Polynomial &holder_polynomial, FluxPointCache const &holder_cache,
      const Polynomial &sharer_polynomial, FluxPointCache const &sharer_cache)
      requires(mini::riemann::ConvectiveDiffusive<Riemann>) {
    Value u_holder = holder_polynomial.GetValue(holder_cache.ijk);
    Value u_sharer = sharer_polynomial.GetValue(sharer_cache.ijk);
    Value f_upwind = riemann.GetFluxUpwind(u_holder, u_sharer);
    auto du_local_holder = holder_polynomial.GetLocalGradient(holder_cache.ijk);
    auto du_local_sharer = sharer_polynomial.GetLocalGradient(sharer_cache.ijk);
    auto du_holder = holder_polynomial.GetGlobalGradient(
        u_holder, du_local_holder, holder_cache.ijk);
    auto du_sharer = sharer_polynomial.GetGlobalGradient(
        u_sharer, du_local_sharer, sharer_cache.ijk);
    auto ddu_holder = holder_polynomial.GetGlobalHessian(
        du_local_holder, holder_cache.ijk);
    auto ddu_sharer = sharer_polynomial.GetGlobalHessian(
        du_local_sharer, sharer_cache.ijk);
    assert(Collinear(holder_cache.normal, sharer_cache.normal));
    const auto &normal = riemann.normal();
    auto du_common = riemann.GetCommonGradient(normal,
        u_holder, u_sharer, du_holder, du_sharer, ddu_holder, ddu_sharer);
    Value u_common = (u_holder + u_sharer) / 2;
    Riemann::MinusViscousFlux(u_common, du_common, normal, &f_upwind);
    auto f_mat_holder = Riemann::GetFluxMatrix(u_holder);
    Riemann::MinusViscousFlux(u_holder, du_holder, &f_mat_holder);
    Value f_holder = f_upwind * holder_cache.scale;
    f_holder -= f_mat_holder * holder_cache.normal;
    auto f_mat_sharer = Riemann::GetFluxMatrix(u_sharer);
    Riemann::MinusViscousFlux(u_sharer, du_sharer, &f_mat_sharer);
    Value f_sharer = f_upwind * (-sharer_cache.scale);
    f_sharer -= f_mat_sharer * sharer_cache.normal;
    return { f_holder, f_sharer };
  }
  void AddFluxOnLocalFaces(Column *residual) const override {
    for (const Face &face : this->part().GetLocalFaces()) {
      const auto &riemanns = this->GetRiemannSolvers(face);
      const auto &holder = face.holder();
      const auto &sharer = face.sharer();
      auto *holder_data = this->AddCellDataOffset(residual, holder.id());
      auto *sharer_data = this->AddCellDataOffset(residual, sharer.id());
      auto const &holder_cache = holder_cache_[face.id()];
      auto &sharer_cache = sharer_cache_[face.id()];
      assert(kFaceQ == face.integrator().CountPoints());
      for (int f = 0; f < kFaceQ; ++f) {
        auto &[holder_solution_points, holder_flux_point] = holder_cache[f];
        auto &[sharer_solution_points, sharer_flux_point] = sharer_cache[f];
        auto [f_holder, f_sharer] = GetFluxOnLocalFace(riemanns[f],
            holder.polynomial(), holder_flux_point,
            sharer.polynomial(), sharer_flux_point);
        for (auto [g_prime, ijk] : holder_solution_points) {
          Value f_correction = f_holder * g_prime;
          Polynomial::MinusValue(f_correction, holder_data, ijk);
        }
        for (auto [g_prime, ijk] : sharer_solution_points) {
          Value f_correction = f_sharer * g_prime;
          Polynomial::MinusValue(f_correction, sharer_data, ijk);
        }
      }
    }
  }
  void AddFluxOnGhostFaces(Column *residual) const override {
    for (const Face &face : this->part().GetGhostFaces()) {
      const auto &riemanns = this->GetRiemannSolvers(face);
      const auto &holder = face.holder();
      const auto &sharer = face.sharer();
      auto *holder_data = this->AddCellDataOffset(residual, holder.id());
      auto const &holder_cache = holder_cache_[face.id()];
      auto &sharer_cache = sharer_cache_[face.id()];
      assert(kFaceQ == face.integrator().CountPoints());
      for (int f = 0; f < kFaceQ; ++f) {
        auto &[holder_solution_points, holder_flux_point] = holder_cache[f];
        auto &[sharer_solution_points, sharer_flux_point] = sharer_cache[f];
        auto [f_holder, _] = GetFluxOnLocalFace(riemanns[f],
            holder.polynomial(), holder_flux_point,
            sharer.polynomial(), sharer_flux_point);
        for (auto [g_prime, ijk] : holder_solution_points) {
          Value f_correction = f_holder * g_prime;
          Polynomial::MinusValue(f_correction, holder_data, ijk);
        }
      }
    }
  }
  void AddFluxOnInviscidWalls(Column *residual) const override {
    for (const auto &name : this->inviscid_wall_) {
      for (const Face &face : this->part().GetBoundaryFaces(name)) {
        const auto &riemanns = this->GetRiemannSolvers(face);
        const auto &holder = face.holder();
        auto *holder_data = this->AddCellDataOffset(residual, holder.id());
        auto const &holder_cache = holder_cache_[face.id()];
        assert(kFaceQ == face.integrator().CountPoints());
        for (int f = 0; f < kFaceQ; ++f) {
          auto &[holder_solution_points, holder_flux_point] = holder_cache[f];
          Value u_holder = holder.polynomial().GetValue(
              holder_flux_point.ijk);
          Value f_upwind = riemanns[f].GetFluxOnInviscidWall(u_holder);
          Value f_holder = f_upwind * holder_flux_point.scale;
          f_holder -=
              Riemann::GetFluxMatrix(u_holder) * holder_flux_point.normal;
          for (auto [g_prime, ijk] : holder_solution_points) {
            Value f_correction = f_holder * g_prime;
            Polynomial::MinusValue(f_correction, holder_data, ijk);
          }
        }
      }
    }
  }
  static Value GetFluxOnNoSlipWall(Riemann const &riemann,
      Value const &wall_value,
      Polynomial const &holder_polynomial, FluxPointCache const &holder_cache)
      requires(!mini::riemann::Diffusive<Riemann>) {
    Value value;
    return value;
  }
  static Value GetFluxOnNoSlipWall(Riemann const &riemann,
      Value const &wall_value,
      Polynomial const &holder_polynomial, FluxPointCache const &holder_cache)
      requires(mini::riemann::ConvectiveDiffusive<Riemann>) {
    Value u_holder = holder_polynomial.GetValue(holder_cache.ijk);
    Value f_upwind = riemann.GetFluxOnInviscidWall(u_holder);
    auto du_local_holder = holder_polynomial.GetLocalGradient(holder_cache.ijk);
    auto du_holder = holder_polynomial.GetGlobalGradient(
        u_holder, du_local_holder, holder_cache.ijk);
    auto f_mat_holder = Riemann::GetFluxMatrix(u_holder);
    Riemann::MinusViscousFlux(u_holder, du_holder, &f_mat_holder);
    const auto &normal = riemann.normal();
    assert(Collinear(normal, holder_cache.normal));
    Scalar value_penalty = riemann.GetValuePenalty();
    Riemann::MinusViscousFluxOnNoSlipWall(wall_value,
        u_holder, du_holder, normal, value_penalty, &f_upwind);
    Value f_holder = f_upwind * holder_cache.scale;
    f_holder -= f_mat_holder * holder_cache.normal;
    return f_holder;
  }
  static Value GetFluxOnSupersonicOutlet(Riemann const &riemann,
      Polynomial const &holder_polynomial, FluxPointCache const &holder_cache)
      requires(!mini::riemann::Diffusive<Riemann>) {
    Value u_holder = holder_polynomial.GetValue(holder_cache.ijk);
    Value f_upwind = riemann.GetFluxOnSupersonicOutlet(u_holder);
    auto f_mat_holder = Riemann::GetFluxMatrix(u_holder);
    Value f_holder = f_upwind * holder_cache.scale;
    f_holder -= f_mat_holder * holder_cache.normal;
    return f_holder;
  }
  static Value GetFluxOnSupersonicOutlet(Riemann const &riemann,
      Polynomial const &holder_polynomial, FluxPointCache const &holder_cache)
      requires(mini::riemann::ConvectiveDiffusive<Riemann>) {
    Value u_holder = holder_polynomial.GetValue(holder_cache.ijk);
    Value f_upwind = riemann.GetFluxOnSupersonicOutlet(u_holder);
    auto du_local_holder = holder_polynomial.GetLocalGradient(holder_cache.ijk);
    auto du_holder = holder_polynomial.GetGlobalGradient(
        u_holder, du_local_holder, holder_cache.ijk);
    const auto &normal = riemann.normal();
    assert(Collinear(normal, holder_cache.normal));
    Riemann::MinusViscousFlux(u_holder, du_holder, normal, &f_upwind);
    auto f_mat_holder = Riemann::GetFluxMatrix(u_holder);
    Riemann::MinusViscousFlux(u_holder, du_holder, &f_mat_holder);
    Value f_holder = f_upwind * holder_cache.scale;
    f_holder -= f_mat_holder * holder_cache.normal;
    assert(f_holder.norm() < 1e-6);
    return f_holder;
  }
  void AddFluxOnSupersonicOutlets(Column *residual) const override {
    for (const auto &name : this->supersonic_outlet_) {
      for (const Face &face : this->part().GetBoundaryFaces(name)) {
        const auto &riemanns = this->GetRiemannSolvers(face);
        const auto &holder = face.holder();
        auto *holder_data = this->AddCellDataOffset(residual, holder.id());
        auto const &holder_cache = holder_cache_[face.id()];
        assert(kFaceQ == face.integrator().CountPoints());
        for (int f = 0; f < kFaceQ; ++f) {
          auto &[holder_solution_points, holder_flux_point] = holder_cache[f];
          auto f_holder = GetFluxOnSupersonicOutlet(riemanns[f],
              holder.polynomial(), holder_flux_point);
          for (auto [g_prime, ijk] : holder_solution_points) {
            Value f_correction = f_holder * g_prime;
            assert(f_correction.norm() < 1e-6);
            assert(0 <= ijk && ijk < kCellQ);
            Polynomial::MinusValue(f_correction, holder_data, ijk);
          }
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
        auto *holder_data = this->AddCellDataOffset(residual, holder.id());
        auto const &holder_cache = holder_cache_[face.id()];
        assert(kFaceQ == face.integrator().CountPoints());
        for (int f = 0; f < kFaceQ; ++f) {
          auto &[holder_solution_points, holder_flux_point] = holder_cache[f];
          Value u_holder = holder.polynomial().GetValue(
              holder_flux_point.ijk);
          Value u_given = func(integrator.GetGlobal(f), this->t_curr_);
          Value f_upwind = riemanns[f].GetFluxOnSupersonicInlet(u_given);
          Value f_holder = f_upwind * holder_flux_point.scale;
          f_holder -=
              Riemann::GetFluxMatrix(u_holder) * holder_flux_point.normal;
          for (auto [g_prime, ijk] : holder_solution_points) {
            Value f_correction = f_holder * g_prime;
            Polynomial::MinusValue(f_correction, holder_data, ijk);
          }
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
        auto *holder_data = this->AddCellDataOffset(residual, holder.id());
        auto const &holder_cache = holder_cache_[face.id()];
        assert(kFaceQ == face.integrator().CountPoints());
        for (int f = 0; f < kFaceQ; ++f) {
          auto &[holder_solution_points, holder_flux_point] = holder_cache[f];
          Value u_holder = holder.polynomial().GetValue(
              holder_flux_point.ijk);
          Value u_given = func(integrator.GetGlobal(f), this->t_curr_);
          Value f_upwind = riemanns[f].GetFluxOnSubsonicInlet(u_holder, u_given);
          Value f_holder = f_upwind * holder_flux_point.scale;
          f_holder -=
              Riemann::GetFluxMatrix(u_holder) * holder_flux_point.normal;
          for (auto [g_prime, ijk] : holder_solution_points) {
            Value f_correction = f_holder * g_prime;
            Polynomial::MinusValue(f_correction, holder_data, ijk);
          }
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
        auto *holder_data = this->AddCellDataOffset(residual, holder.id());
        auto const &holder_cache = holder_cache_[face.id()];
        assert(kFaceQ == face.integrator().CountPoints());
        for (int f = 0; f < kFaceQ; ++f) {
          auto &[holder_solution_points, holder_flux_point] = holder_cache[f];
          Value u_holder = holder.polynomial().GetValue(
              holder_flux_point.ijk);
          Value u_given = func(integrator.GetGlobal(f), this->t_curr_);
          Value f_upwind = riemanns[f].GetFluxOnSubsonicOutlet(u_holder, u_given);
          Value f_holder = f_upwind * holder_flux_point.scale;
          f_holder -=
              Riemann::GetFluxMatrix(u_holder) * holder_flux_point.normal;
          for (auto [g_prime, ijk] : holder_solution_points) {
            Value f_correction = f_holder * g_prime;
            Polynomial::MinusValue(f_correction, holder_data, ijk);
          }
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
        auto *holder_data = this->AddCellDataOffset(residual, holder.id());
        auto const &holder_cache = holder_cache_[face.id()];
        assert(kFaceQ == face.integrator().CountPoints());
        for (int f = 0; f < kFaceQ; ++f) {
          auto &[holder_solution_points, holder_flux_point] = holder_cache[f];
          Value u_holder = holder.polynomial().GetValue(
              holder_flux_point.ijk);
          Value u_given = func(integrator.GetGlobal(f), this->t_curr_);
          Value f_upwind = riemanns[f].GetFluxOnSmartBoundary(u_holder, u_given);
          Value f_holder = f_upwind * holder_flux_point.scale;
          f_holder -=
              Riemann::GetFluxMatrix(u_holder) * holder_flux_point.normal;
          for (auto [g_prime, ijk] : holder_solution_points) {
            Value f_correction = f_holder * g_prime;
            Polynomial::MinusValue(f_correction, holder_data, ijk);
          }
        }
      }
    }
  }
};

}  // namespace fr
}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_FR_GENERAL_HPP_
