// Copyright 2024 PEI Weicheng

#include <concepts>
#include <fstream>

#include "mini/algebra/eigen.hpp"
#include "mini/mesh/vtk.hpp"

#include <CGAL/Simple_cartesian.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/Projection_traits_xy_3.h>
#include <CGAL/Delaunay_triangulation_2.h>

template <std::floating_point Real>
void WriteVtu(std::string const &filename, bool binary,
    int n_point, Real const *x, Real const *y,  Real const *z,
    int n_cell) {
  std::string endianness
      = (std::endian::native == std::endian::little)
      ? "\"LittleEndian\"" : "\"BigEndian\"";
  auto format = binary ? "\"binary\"" : "\"ascii\"";
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
  for (int i = 0; i < n_cell; ++i) {
  }
  vtu << "\n        </DataArray>\n";
  // Write cell connectivity offsets:
  vtu << "        <DataArray type=\"Int32\" Name=\"offsets\" "
      << "format=" << format << ">\n";
  int offset = 0;
  for (int i = 0; i < n_cell; ++i) {
    offset += 3;
    vtu << offset << ' ';
  }
  vtu << "\n        </DataArray>\n";
  // Write cell types:
  vtu << "        <DataArray type=\"UInt8\" Name=\"types\" "
      << "format=" << format << ">\n";
  for (int i = 0; i < n_cell; ++i) {
    vtu << static_cast<int>(mini::mesh::vtk::CellType::kTetrahedron4) << ' ';
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

  // Triangulate the points.
  using Kernel = CGAL::Simple_cartesian<Real>;
  using Point = Kernel::Point_3;
  using GeoTraits = CGAL::Projection_traits_xy_3<Kernel>;
  using Delaunay = CGAL::Delaunay_triangulation_2<GeoTraits>;

  auto delaunay = Delaunay();
  for (int i = 0; i < n_point; i++) {
    delaunay.insert(Point(x[i], y[i], z[i]));
  }
  std::cout << delaunay.number_of_vertices() << std::endl;
  std::cout << delaunay.number_of_faces() << std::endl;
  std::cout << delaunay << std::endl;

  // Write the points and triangles.
  WriteVtu("test.vtu", false, n_point, x.data(), y.data(), z.data(), 0);
  return 0;
}
