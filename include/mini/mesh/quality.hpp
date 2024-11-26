// Copyright 2021 PEI Weicheng and YANG Minghao and JIANG Yuyan
#ifndef MINI_MESH_QUALITY_HPP_
#define MINI_MESH_QUALITY_HPP_

#include <concepts>

#include "cgnslib.h"

#include "mini/mesh/cgns.hpp"
#include "mini/coordinate/hexahedron.hpp"
#include "mini/integrator/hexahedron.hpp"
#include "mini/integrator/lobatto.hpp"

namespace mini {
namespace mesh {

template <std::floating_point Real>
class QualityMeasure {
 public:
  using File = mini::mesh::cgns::File<Real>;
  using Zone = mini::mesh::cgns::Zone<Real>;
  using Coordinates = mini::mesh::cgns::Coordinates<Real>;
  using Solution = mini::mesh::cgns::Solution<Real>;
  using Field = mini::mesh::cgns::Field<Real>;

  using Gx = mini::integrator::Lobatto<Real, 3>;
  using Integrator = mini::integrator::Hexahedron<Gx, Gx, Gx>;
  using Global = typename Integrator::Global;

  static Global GetGlobal(Coordinates const &coordinates, cgsize_t i_node/* 1-based */) {
    return { coordinates.x(i_node), coordinates.y(i_node), coordinates.z(i_node) };
  }

  static Integrator BuildHexa8(Coordinates const &coordinates, cgsize_t const *nodes) {
    auto coordinate = mini::coordinate::Hexahedron8<Real>{
        GetGlobal(coordinates, nodes[0]), GetGlobal(coordinates, nodes[1]),
        GetGlobal(coordinates, nodes[2]), GetGlobal(coordinates, nodes[3]),
        GetGlobal(coordinates, nodes[4]), GetGlobal(coordinates, nodes[5]),
        GetGlobal(coordinates, nodes[6]), GetGlobal(coordinates, nodes[7]) };
    return Integrator(coordinate);
  }

  static Integrator BuildHexa27(Coordinates const &coordinates, cgsize_t const *nodes) {
    auto coordinate = mini::coordinate::Hexahedron27<Real>{
        GetGlobal(coordinates, nodes[0]), GetGlobal(coordinates, nodes[1]),
        GetGlobal(coordinates, nodes[2]), GetGlobal(coordinates, nodes[3]),
        GetGlobal(coordinates, nodes[4]), GetGlobal(coordinates, nodes[5]),
        GetGlobal(coordinates, nodes[6]), GetGlobal(coordinates, nodes[7]),
        GetGlobal(coordinates, nodes[8]), GetGlobal(coordinates, nodes[9]),
        GetGlobal(coordinates, nodes[10]), GetGlobal(coordinates, nodes[11]),
        GetGlobal(coordinates, nodes[12]), GetGlobal(coordinates, nodes[13]),GetGlobal(coordinates, nodes[14]), GetGlobal(coordinates, nodes[15]),
        GetGlobal(coordinates, nodes[16]), GetGlobal(coordinates, nodes[17]),
        GetGlobal(coordinates, nodes[18]), GetGlobal(coordinates, nodes[19]),
        GetGlobal(coordinates, nodes[20]), GetGlobal(coordinates, nodes[21]),
        GetGlobal(coordinates, nodes[22]), GetGlobal(coordinates, nodes[23]),
        GetGlobal(coordinates, nodes[24]), GetGlobal(coordinates, nodes[25]),
        GetGlobal(coordinates, nodes[26]) };
    return Integrator(coordinate);
  }

  static auto *SelectBuilder(CGNS_ENUMT(ElementType_t) type) {
    Integrator (*ptr)(Coordinates const &, cgsize_t const *);
    switch (type) {
    case CGNS_ENUMV(HEXA_8):
      ptr = BuildHexa8;
      break;
    case CGNS_ENUMV(HEXA_27):
      ptr = BuildHexa27;
      break;
    default:
      ptr = nullptr;
    }
    return ptr;
  }
};

}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_QUALITY_HPP_
