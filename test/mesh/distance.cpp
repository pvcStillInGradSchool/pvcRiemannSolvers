// Copyright 2024 PEI Weicheng

#include <concepts>
#include <fstream>
#include <unordered_map>
#include <vector>

#include "mini/algebra/eigen.hpp"
#include "mini/mesh/vtk.hpp"

#include <CGAL/Simple_cartesian.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/Projection_traits_xy_3.h>
#include <CGAL/Delaunay_triangulation_2.h>

template <typename Delaunay>
std::vector<std::array<int, 3>> GetFaces(Delaunay const &delaunay) {
  // prepare the vertex_handle-to-vertex_index map:
  auto vertex_map = std::unordered_map<
      typename Delaunay::Vertex_handle, int>();
  int i = 0;  // index of vertex
  for (auto vertex_handle : delaunay.finite_vertex_handles()) {
    vertex_map[vertex_handle] = i++;
  }
  assert(delaunay.number_of_vertices() == i);
  // build faces (triangles)
  std::vector<std::array<int, 3>> faces;
  faces.reserve(delaunay.number_of_faces());
  for (auto face_handle : delaunay.finite_face_handles()) {
    auto &face = faces.emplace_back();
    face[0] = vertex_map.at(face_handle->vertex(0));
    face[1] = vertex_map.at(face_handle->vertex(1));
    face[2] = vertex_map.at(face_handle->vertex(2));
  }
  assert(faces.size() == delaunay.number_of_faces());
  return faces;
}

std::vector<std::array<int, 2>> GetEdges(std::vector<std::array<int, 3>> const &faces) {
  auto edges = std::vector<std::array<int, 2>>();
  for (auto [a, b, c] : faces) {
    auto emplace_back = [&edges](int i, int j) {
      auto &edge = edges.emplace_back();
      edge[0] = std::min(i, j);
      edge[1] = std::max(i, j);
    };
    emplace_back(a, b); emplace_back(b, c); emplace_back(c, a);
  }
  std::ranges::sort(edges);
  auto ret = std::ranges::unique(edges);
  edges.erase(ret.begin(), ret.end());
  return edges;
}

template <std::floating_point Real, int kNodes>
void WriteVtu(std::string const &filename, bool binary,
    int n_point, Real const *x, Real const *y,  Real const *z,
    std::vector<std::array<int, kNodes>> const &cells,
    mini::mesh::vtk::CellType vtk_cell_type) {
  std::string endianness
      = (std::endian::native == std::endian::little)
      ? "\"LittleEndian\"" : "\"BigEndian\"";
  auto format = binary ? "\"binary\"" : "\"ascii\"";
  int n_cell = cells.size();
  assert(kNodes == mini::mesh::vtk::CountNodes(vtk_cell_type));
  // Initialize the VTU file:
  auto vtu = std::ofstream(filename,
      std::ios::out | (binary ? (std::ios::binary) : std::ios::out));
  vtu << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\""
      << " byte_order=" << endianness << " header_type=\"UInt64\">\n";
  vtu << "  <UnstructuredGrid>\n";
  vtu << "    <Piece NumberOfPoints=\"" << n_point
      << "\" NumberOfCells=\"" << n_cell << "\">\n";
  // Write point coordinates:
  vtu << "      <Points>\n";
  vtu << "        <DataArray type=\"Float64\" Name=\"Points\" "
      << "NumberOfComponents=\"3\" format=" << format << ">\n";
  for (int i = 0; i < n_point; ++i) {
    vtu << x[i] << ' ' << y[i] << ' ' << z[i] << ' ';
  }
  vtu << "\n        </DataArray>\n";
  vtu << "      </Points>\n";
  vtu << "      <Cells>\n";
  // Write cell connectivities:
  vtu << "        <DataArray type=\"Int32\" Name=\"connectivity\" "
      << "format=" << format << ">\n";
  for (auto &cell : cells) {
    for (int i = 0; i < kNodes; ++i) {
      vtu << cell[i] << ' ';
    }
  }
  vtu << "\n        </DataArray>\n";
  // Write cell connectivity offsets:
  vtu << "        <DataArray type=\"Int32\" Name=\"offsets\" "
      << "format=" << format << ">\n";
  int offset = 0;
  for (int i = 0; i < n_cell; ++i) {
    offset += kNodes;
    vtu << offset << ' ';
  }
  vtu << "\n        </DataArray>\n";
  // Write cell types:
  vtu << "        <DataArray type=\"UInt8\" Name=\"types\" "
      << "format=" << format << ">\n";
  for (int i = 0; i < n_cell; ++i) {
    vtu << static_cast<int>(vtk_cell_type) << ' ';
  }
  vtu << "\n        </DataArray>\n";
  vtu << "      </Cells>\n";
  vtu << "    </Piece>\n";
  vtu << "  </UnstructuredGrid>\n";
  vtu << "</VTKFile>\n";
}

int main(int argc, char *argv[]) {
  using Real = double;
  // ./distance <n_point>
  int n_point = std::atoi(argv[1]);

  // Build random points in \f$ [-1, 1]^2 \times 0 \f$
  mini::algebra::DynamicVector<Real> x(n_point), y(n_point), z(n_point);
  x.setRandom(); y.setRandom(); z.setZero();
  // Fix corner points:
  x[0] = x[2] = y[0] = y[1] = -1.;
  x[1] = x[3] = y[2] = y[3] = +1.;

  // Triangulate the points.
  using Kernel = CGAL::Simple_cartesian<Real>;
  using Point = Kernel::Point_3;
  using GeoTraits = CGAL::Projection_traits_xy_3<Kernel>;
  using Delaunay = CGAL::Delaunay_triangulation_2<GeoTraits>;

  auto delaunay = Delaunay();
  for (int i = 0; i < n_point; i++) {
    delaunay.insert(Point(x[i], y[i], z[i]));
  }

  auto faces = GetFaces(delaunay);
  auto edges = GetEdges(faces);

  // Write the points and triangles.
  WriteVtu<Real, 3>("cells.vtu", false, n_point, x.data(), y.data(), z.data(),
      faces, mini::mesh::vtk::CellType::kTriangle3);
  WriteVtu<Real, 2>("edges.vtu", false, n_point, x.data(), y.data(), z.data(),
      edges, mini::mesh::vtk::CellType::kLine2);
  return 0;
}
