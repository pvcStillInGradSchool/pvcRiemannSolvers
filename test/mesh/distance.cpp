// Copyright 2024 PEI Weicheng

#include "mini/algebra/eigen.hpp"
#include "mini/mesh/cgns.hpp"

#include <CGAL/Simple_cartesian.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/Projection_traits_xy_3.h>
#include <CGAL/Delaunay_triangulation_2.h>

int main(int argc, char *argv[]) {
  using Real = double;
  // ./distance <n_point>
  int n_point = std::atoi(argv[1]);

  // Build random points in \f$ [-1, 1]^2 \times 0 \f$
  mini::algebra::DynamicVector<Real> x(n_point), y(n_point), z(n_point);
  x.setRandom(); y.setRandom(); z.setZero();

  // Write the points.


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
  return 0;
}
