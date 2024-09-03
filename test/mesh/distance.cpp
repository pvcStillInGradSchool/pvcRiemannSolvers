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

/**
 * @brief The distance function of a rectangle.
 * 
 */
template <std::floating_point Real>
Real Rectangle(Real x, Real y, Real x_min, Real x_max, Real y_min, Real y_max) {
  return -std::min(std::min(y - y_min, y_max - y),
                   std::min(x - x_min, x_max - x));
}

/**
 * @brief The distance function of a circle.
 * 
 */
template <std::floating_point Real>
Real Circle(Real x, Real y, Real x_center, Real y_center, Real radius) {
  return std::hypot(x - x_center, y - y_center) - radius;
}

/**
 * @brief The distance function of \f$ A \setminus B \f$.
 * 
 */
template <std::floating_point Real>
auto Difference(Real a, Real b) {
  return std::max(a, -b);
}

template <class Vector, class Distance>
int RejectPoints(Vector *x, Vector *y, Distance &&distance) {
  int n = x->size();
  assert(n == y->size());
  auto valid = std::vector<int>();
  for (int i = 0; i < n; i++) {
    if (distance((*x)[i], (*y)[i]) <= 0) {
      valid.push_back(i);
    }
  }
  int n_new = valid.size();
  Vector x_new(n_new), y_new(n_new);
  for (int i_new = 0; i_new < n_new; i_new++) {
    int i = valid[i_new];
    x_new[i_new] = (*x)[i];
    y_new[i_new] = (*y)[i];
  }
  std::swap(x_new, *x);
  std::swap(y_new, *y);
  return n_new;
}

template <class Vector, class Distance>
int RejectFaces(std::vector<std::array<int, 3>> *faces,
    Vector const &x, Vector const &y, Distance &&distance,
    decltype(x[0]) eps) {
  int n = faces->size();
  auto valid = std::vector<int>();
  for (int i = 0; i < n; i++) {
    auto [a, b, c] = faces->at(i);
    auto x_center = (x[a] + x[b] + x[c]) / 3.;
    auto y_center = (y[a] + y[b] + y[c]) / 3.;
    if (distance(x_center, y_center) + eps <= 0.) {
      valid.push_back(i);
    }
  }
  int n_new = valid.size();
  std::vector<std::array<int, 3>> faces_new(n_new);
  for (int i_new = 0; i_new < n_new; i_new++) {
    faces_new[i_new] = faces->at(valid[i_new]);
  }
  std::swap(faces_new, *faces);
  return n_new;
}

template <std::floating_point Real, int kNodes, class Distance>
void WriteVtu(std::string const &filename, bool binary,
    int n_point, Real const *x, Real const *y,  Real const *z,
    std::vector<std::array<int, kNodes>> const &cells,
    mini::mesh::vtk::CellType vtk_cell_type,
    Distance &&distance) {
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
  // Write the value of distance(x, y) as PointData:
  vtu << "      <PointData>\n";
  vtu << "        <DataArray type=\"Float64\" Name=\""
      << "DistanceToBoundary" << "\" format=" << format << ">\n";
  for (int i = 0; i < n_point; ++i) {
    vtu << distance(x[i], y[i]) << ' ';
  }
  vtu << "\n        </DataArray>\n";
  vtu << "      </PointData>\n";
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

  Real const h_0 = 0.005;
  Real const eps = 0.005;
  auto g_eps = h_0 * 0.001;  // for rejecting out-of-domain faces
  auto d_eps = h_0 * std::sqrt(eps);  // for finite-differencing d(x, y)

  // Build random points in \f$ [-1, 1]^2 \times 0 \f$
  mini::algebra::DynamicVector<Real> x(n_point), y(n_point), z(n_point);
  x.setRandom(); y.setRandom(); z.setZero();
  // Fix corner points:
  int n_fixed = 4;
  Real x_center = 0.0, y_center = 0.0, radius = 0.5;
  Real x_min = x_center - 1.0;
  Real x_max = -x_min;
  Real y_min = y_center - 1.0;
  Real y_max = -y_min;
  x[0] = x[2] = x_min;
  x[1] = x[3] = x_max;
  y[0] = y[1] = y_min;
  y[2] = y[3] = y_max;

  auto distance = [&](Real a, Real b) {
    return Difference(
        Rectangle(a, b, x_min, x_max, y_min, y_max),
        Circle(a, b, x_center, y_center, radius));
  };

  auto scaling = [&](Real a, Real b) {
    return 0.05 + 0.3 * Circle(a, b, x_center, y_center, radius);
  };

  // Reject out-of-domain (d > 0) points:
  n_point = RejectPoints(&x, &y, distance);
  assert(n_point == x.size());
  assert(n_point == y.size());

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
  int n_face = RejectFaces(&faces, x, y, distance, g_eps);
  assert(n_face <= delaunay.number_of_faces());
  auto edges = GetEdges(faces);
  int n_edge = edges.size();

  // Build actual and expect lengths:
  mini::algebra::DynamicVector<Real> bar_x(n_edge), bar_y(n_edge),
      center_x(n_edge), center_y(n_edge),
      actual_l(n_edge), expect_l(n_edge);
  for (int i = 0; i < n_edge; i++) {
    auto [u, v] = edges[i];
    Real x_u = x[u], x_v = x[v];
    Real y_u = y[u], y_v = y[v];
    actual_l[i] = std::hypot(bar_x[i] = x_v - x_u, bar_y[i] = y_v - y_u);
    expect_l[i] = scaling((x_u + x_v) / 2, (y_u + y_v) / 2);
  }
  expect_l *= 1.2 * std::sqrt(actual_l.squaredNorm() / expect_l.squaredNorm());

  // Get forces at nodes:
  mini::algebra::DynamicVector<Real> force_x(n_point), force_y(n_point);
  for (int i = 0; i < n_edge; i++) {
    // repulsive force for compressed bars
    Real force = /* 1.0 * */std::max(0., expect_l[i] - actual_l[i]);
    auto [u, v] = edges[i];
    force_x[u] = -(force_x[v] = force * bar_x[i]);
    force_y[u] = -(force_y[v] = force * bar_y[i]);
  }
  // No force at fixed points:
  for (int i = 0; i < n_fixed; i++) {
    force_x[i] = force_y[i] = 0.;
  }

  // Move points:
  Real delta_t = 0.2;
  mini::algebra::DynamicVector<Real> shift_x(n_point), shift_y(n_point);
  for (int i = n_fixed; i < n_point; i++) {
    x[i] += (shift_x[i] = delta_t * force_x[i]);
    y[i] += (shift_y[i] = delta_t * force_y[i]);
  }

  // Project back out-of-domain points:
  auto out = std::vector<int>();
  for (int i = n_fixed; i < n_point; i++) {
    Real d = distance(x[i], y[i]);
    if (d <= 0) {
      continue;
    }
    // Numerical gradient of d(x, y):
    Real grad_x = (distance(x[i] + d_eps, y[i]) - d) / d_eps;
    Real grad_y = (distance(x[i], y[i] + d_eps) - d) / d_eps;
    x[i] -= grad_x * d;
    y[i] -= grad_y * d;
    shift_x[i] = shift_y[i] = 0.;
  }

  // Write the points and triangles.
  WriteVtu<Real, 3>("cells.vtu", false, n_point, x.data(), y.data(), z.data(),
      faces, mini::mesh::vtk::CellType::kTriangle3, distance);
  WriteVtu<Real, 2>("edges.vtu", false, n_point, x.data(), y.data(), z.data(),
      edges, mini::mesh::vtk::CellType::kLine2, distance);
  return 0;
}
