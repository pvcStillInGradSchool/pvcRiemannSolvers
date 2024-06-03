// Copyright 2023 PEI Weicheng
#ifndef MINI_SPATIAL_FR_LOBATTO_HPP_
#define MINI_SPATIAL_FR_LOBATTO_HPP_

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

#include "mini/polynomial/hexahedron.hpp"
#include "mini/integrator/lobatto.hpp"
#include "mini/spatial/fr/general.hpp"
#include "mini/basis/vincent.hpp"

namespace mini {
namespace spatial {
namespace fr {

/**
 * @brief A specialized version of FR using a Lagrange expansion on Lobatto roots with the "Lumping Lobatto" correction function.
 * 
 * The \f$ g_\mathrm{right} \f$ only corrects the flux divergence at the rightest solution point, which is a flux point.
 * 
 * @tparam Part 
 * @tparam Riem 
 */
template <typename P, typename R>
class Lobatto : public General<P, R> {
 public:
  using Base = General<P, R>;
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
  using Vincent = typename Base::Vincent;

 protected:
  using FluxMatrix = typename Riemann::FluxMatrix;

  static constexpr int kLineQ = polynomial::LineIntegrator<Polynomial>::Q;
  static constexpr int kFaceQ = kLineQ * kLineQ;
  static constexpr int kCellQ = kLineQ * kFaceQ;

  struct LineCache : Base::FluxPointCache {
    Scalar g_prime;
  };

  using FaceCache = std::array<LineCache, kFaceQ>;
  std::vector<FaceCache> holder_cache_;
  std::vector<FaceCache> sharer_cache_;

  static bool Collinear(Global const &a, Global const &b) {
    return std::abs(1 - std::abs(a.dot(b) / a.norm() / b.norm())) < 1e-8;
  }

  template <std::ranges::input_range Range, class FaceToCell>
  void CacheCorrectionGradients(Range &&faces, FaceToCell &&face_to_cell,
      std::vector<FaceCache> *cache) {
    Scalar g_prime = this->vincent_.LocalToRightDerivative(1);
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
        auto &flux_point = curr_face.at(f);
        auto &flux_point_coord = face_integrator.GetGlobal(f);
        auto [i, j, k] = cell_polynomial.FindCollinearIndex(flux_point_coord, i_face);
        switch (i_face) {
        case 0:
          assert(k == -1);
          flux_point.ijk = cell_basis.index(i, j, 0);
          assert(Near(flux_point_coord, cell_integrator.GetGlobal(flux_point.ijk)));
          flux_point.normal =
              cell_polynomial.GetJacobianAssociated(flux_point.ijk).col(Z);
          assert(Collinear(face_normal, flux_point.normal));
          flux_point.scale = -flux_point.normal.norm();
          flux_point.g_prime = -g_prime;
          break;
        case 1:
          assert(j == -1);
          flux_point.ijk = cell_basis.index(i, 0, k);
          assert(Near(flux_point_coord, cell_integrator.GetGlobal(flux_point.ijk)));
          flux_point.normal =
              cell_polynomial.GetJacobianAssociated(flux_point.ijk).col(Y);
          assert(Collinear(face_normal, flux_point.normal));
          flux_point.scale = -flux_point.normal.norm();
          flux_point.g_prime = -g_prime;
          break;
        case 2:
          assert(i == -1);
          flux_point.ijk = cell_basis.index(kLineQ - 1, j, k);
          assert(Near(flux_point_coord, cell_integrator.GetGlobal(flux_point.ijk)));
          flux_point.normal =
              cell_polynomial.GetJacobianAssociated(flux_point.ijk).col(X);
          assert(Collinear(face_normal, flux_point.normal));
          flux_point.scale = +flux_point.normal.norm();
          flux_point.g_prime = +g_prime;
          break;
        case 3:
          assert(j == -1);
          flux_point.ijk = cell_basis.index(i, kLineQ - 1, k);
          assert(Near(flux_point_coord, cell_integrator.GetGlobal(flux_point.ijk)));
          flux_point.normal =
              cell_polynomial.GetJacobianAssociated(flux_point.ijk).col(Y);
          assert(Collinear(face_normal, flux_point.normal));
          flux_point.scale = +flux_point.normal.norm();
          flux_point.g_prime = +g_prime;
          break;
        case 4:
          assert(i == -1);
          flux_point.ijk = cell_basis.index(0, j, k);
          assert(Near(flux_point_coord, cell_integrator.GetGlobal(flux_point.ijk)));
          flux_point.normal =
              cell_polynomial.GetJacobianAssociated(flux_point.ijk).col(X);
          assert(Collinear(face_normal, flux_point.normal));
          flux_point.scale = -flux_point.normal.norm();
          flux_point.g_prime = -g_prime;
          break;
        case 5:
          assert(k == -1);
          flux_point.ijk = cell_basis.index(i, j, kLineQ - 1);
          assert(Near(flux_point_coord, cell_integrator.GetGlobal(flux_point.ijk)));
          flux_point.normal =
              cell_polynomial.GetJacobianAssociated(flux_point.ijk).col(Z);
          assert(Collinear(face_normal, flux_point.normal));
          flux_point.scale = +flux_point.normal.norm();
          flux_point.g_prime = +g_prime;
          break;
        default:
          assert(false);
        }
      }
    }
  }

  void CheckCacheConsistency(std::vector<FaceCache> const &this_cache,
      std::vector<typename Base::FaceCache> const &base_cache) const {
    int n = this_cache.size();
    for (int i = 0; i < n; ++i) {
      auto &face_cache = this_cache.at(i);
      auto &face_cache_base = base_cache.at(i);
      for (int f = 0; f < kFaceQ; ++f) {
        auto &flux_point = face_cache[f];
        auto &[solution_points, flux_point_base] = face_cache_base[f];
        assert(flux_point.ijk == flux_point_base.ijk);
        assert(flux_point.scale == flux_point_base.scale);
        assert(flux_point.normal == flux_point_base.normal);
        int n_match = 0;
        assert(kLineQ == solution_points.size());
        for (int g = 0; g < kLineQ; ++g) {
          if (flux_point.ijk == solution_points[g].ijk) {
            n_match++;
            assert(solution_points[g].g_prime == flux_point.g_prime);
          } else {
            assert(std::abs(solution_points[g].g_prime) < 1e-8);
          }
        }
        assert(n_match == 1);
      }
    }
  }

 public:
  explicit Lobatto(Part *part_ptr)
      : Base(part_ptr, Vincent::HuynhLumpingLobatto(Part::kDegrees)) {
    // TODO(PVC): remove duplicated code
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
    CheckCacheConsistency(holder_cache_, this->Base::holder_cache_);
    CheckCacheConsistency(sharer_cache_, this->Base::sharer_cache_);
  }
  Lobatto(const Lobatto &) = default;
  Lobatto &operator=(const Lobatto &) = default;
  Lobatto(Lobatto &&) noexcept = default;
  Lobatto &operator=(Lobatto &&) noexcept = default;
  ~Lobatto() noexcept = default;

  std::string name() const override {
    return "FR::Lobatto";
  }

 protected:  // override virtual methods defined in Base
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
        auto &holder_flux_point = holder_cache[f];
        auto &sharer_flux_point = sharer_cache[f];
        auto [f_holder, f_sharer] = Base::GetFluxOnLocalFace(riemanns[f],
            riemanns[f].normal().dot(face.HolderToSharer()),
            holder.polynomial(), holder_flux_point,
            sharer.polynomial(), sharer_flux_point);
        f_holder *= holder_flux_point.g_prime;
        Polynomial::MinusValue(f_holder, holder_data, holder_flux_point.ijk);
        f_sharer *= sharer_flux_point.g_prime;
        Polynomial::MinusValue(f_sharer, sharer_data, sharer_flux_point.ijk);
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
        auto &holder_flux_point = holder_cache[f];
        auto &sharer_flux_point = sharer_cache[f];
        auto [f_holder, _] = Base::GetFluxOnLocalFace(riemanns[f],
            riemanns[f].normal().dot(face.HolderToSharer()),
            holder.polynomial(), holder_flux_point,
            sharer.polynomial(), sharer_flux_point);
        f_holder *= holder_flux_point.g_prime;
        Polynomial::MinusValue(f_holder, holder_data, holder_flux_point.ijk);
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
          auto &holder_flux_point = holder_cache[f];
          Value u_holder = holder.polynomial().GetValue(
              holder_flux_point.ijk);
          Value f_upwind = riemanns[f].GetFluxOnInviscidWall(u_holder);
          Value f_holder = f_upwind * holder_flux_point.scale;
          f_holder -=
              Riemann::GetFluxMatrix(u_holder) * holder_flux_point.normal;
          f_holder *= holder_flux_point.g_prime;
          Polynomial::MinusValue(f_holder, holder_data, holder_flux_point.ijk);
        }
      }
    }
  }

 public:
  void AddFluxOnNoSlipWalls(Column *residual) const override {
    for (const auto &[name, func] : this->no_slip_wall_) {
      for (const Face &face : this->part().GetBoundaryFaces(name)) {
        const auto &riemanns = this->GetRiemannSolvers(face);
        const auto &holder = face.holder();
        auto *holder_data = this->AddCellDataOffset(residual, holder.id());
        auto const &holder_cache = holder_cache_[face.id()];
        auto const &integrator = face.integrator();
        auto const &direction = face.HolderToSharer();
        assert(kFaceQ == integrator.CountPoints());
        for (int f = 0; f < kFaceQ; ++f) {
          auto &holder_flux_point = holder_cache[f];
          Value wall_value = func(integrator.GetGlobal(f), this->t_curr_);
          Value f_holder = Base::GetFluxOnNoSlipWall(riemanns[f],
              direction.dot(riemanns[f].normal()),
              wall_value, holder.polynomial(), holder_flux_point);
          f_holder *= holder_flux_point.g_prime;
          Polynomial::MinusValue(f_holder, holder_data, holder_flux_point.ijk);
        }
      }
    }
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
          auto &holder_flux_point = holder_cache[f];
          auto f_holder = Base::GetFluxOnSupersonicOutlet(riemanns[f],
              holder.polynomial(), holder_flux_point);
          f_holder *= holder_flux_point.g_prime;
          assert(f_holder.norm() < 1e-6);
          assert(0 <= holder_flux_point.ijk && holder_flux_point.ijk < kCellQ);
          Polynomial::MinusValue(f_holder, holder_data, holder_flux_point.ijk);
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
          auto &holder_flux_point = holder_cache[f];
          Value u_holder = holder.polynomial().GetValue(
              holder_flux_point.ijk);
          Value u_given = func(integrator.GetGlobal(f), this->t_curr_);
          Value f_upwind = riemanns[f].GetFluxOnSupersonicInlet(u_given);
          Value f_holder = f_upwind * holder_flux_point.scale;
          f_holder -=
              Riemann::GetFluxMatrix(u_holder) * holder_flux_point.normal;
          f_holder *= holder_flux_point.g_prime;
          Polynomial::MinusValue(f_holder, holder_data, holder_flux_point.ijk);
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
          auto &holder_flux_point = holder_cache[f];
          Value u_holder = holder.polynomial().GetValue(
              holder_flux_point.ijk);
          Value u_given = func(integrator.GetGlobal(f), this->t_curr_);
          Value f_upwind = riemanns[f].GetFluxOnSubsonicInlet(u_holder, u_given);
          Value f_holder = f_upwind * holder_flux_point.scale;
          f_holder -=
              Riemann::GetFluxMatrix(u_holder) * holder_flux_point.normal;
          f_holder *= holder_flux_point.g_prime;
          Polynomial::MinusValue(f_holder, holder_data, holder_flux_point.ijk);
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
          auto &holder_flux_point = holder_cache[f];
          Value u_holder = holder.polynomial().GetValue(
              holder_flux_point.ijk);
          Value u_given = func(integrator.GetGlobal(f), this->t_curr_);
          Value f_upwind = riemanns[f].GetFluxOnSubsonicOutlet(u_holder, u_given);
          Value f_holder = f_upwind * holder_flux_point.scale;
          f_holder -=
              Riemann::GetFluxMatrix(u_holder) * holder_flux_point.normal;
          f_holder *= holder_flux_point.g_prime;
          Polynomial::MinusValue(f_holder, holder_data, holder_flux_point.ijk);
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
          auto &holder_flux_point = holder_cache[f];
          Value u_holder = holder.polynomial().GetValue(
              holder_flux_point.ijk);
          Value u_given = func(integrator.GetGlobal(f), this->t_curr_);
          Value f_upwind = riemanns[f].GetFluxOnSmartBoundary(u_holder, u_given);
          Value f_holder = f_upwind * holder_flux_point.scale;
          f_holder -=
              Riemann::GetFluxMatrix(u_holder) * holder_flux_point.normal;
          f_holder *= holder_flux_point.g_prime;
          Polynomial::MinusValue(f_holder, holder_data, holder_flux_point.ijk);
        }
      }
    }
  }
};

}  // namespace fr
}  // namespace spatial
}  // namespace mini

#endif  // MINI_SPATIAL_FR_LOBATTO_HPP_
