// Copyright 2024 PEI Weicheng

#include <algorithm>
#include <concepts>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "mini/algebra/eigen.hpp"
#include "mini/mesh/vtk.hpp"

#include <CGAL/Simple_cartesian.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/Projection_traits_xy_3.h>
#include <CGAL/Delaunay_triangulation_2.h>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "yaml-cpp/yaml.h"

#define NSTREAMS 4

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
__device__ __host__
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

int n_fixed = 4;
double x_center = 0.0, y_center = 0.0, radius = 0.5;
double x_min = x_center - 1.0;
double x_max = -x_min;
double y_min = y_center - 1.0;
double y_max = -y_min;

template <std::floating_point Real>
Real distance(Real a, Real b) {
  return Difference(
      Rectangle(a, b, x_min, x_max, y_min, y_max),
      Circle(a, b, x_center, y_center, radius));
}

template <std::floating_point Real>
__device__ __host__
Real scaling(Real a, Real b) {
  return 0.05 + 0.3 * Circle(a, b, 0.0, 0.0, 0.5);
};

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

template <std::floating_point Real>
void WriteFrame(int i_frame, int n_point,
    Real const *x, Real const *y, Real const *z,
    std::vector<std::array<int, 3>> const &faces) {
  auto vtu_name = std::string("Frame");
  vtu_name += std::to_string(i_frame);
  vtu_name += ".vtu";
  WriteVtu<Real, 3>(vtu_name, false, n_point, x, y, z,
      faces, mini::mesh::vtk::CellType::kTriangle3, distance<Real>);
  std::cout << vtu_name << " written\n";
}

template <std::floating_point Real>
using HostDynamicVector = mini::algebra::DynamicVector<Real>;

template <std::floating_point Real>
class Memory {
 public:
  Real *x_u{nullptr}, *x_v{nullptr}, *y_u{nullptr}, *y_v{nullptr};
  Real *bar_x{nullptr}, *bar_y{nullptr}, *actual_l{nullptr}, *expect_l{nullptr};

  template <std::invocable<void **, int> M>
  void Malloc(int n_byte, M &&malloc) {
    malloc((void **)&x_u, n_byte);
    malloc((void **)&x_v, n_byte);
    malloc((void **)&y_u, n_byte);
    malloc((void **)&y_v, n_byte);
    malloc((void **)&bar_x, n_byte);
    malloc((void **)&bar_y, n_byte);
    malloc((void **)&actual_l, n_byte);
    malloc((void **)&expect_l, n_byte);
  }

  template <std::invocable<void *> F>
  void Free(F &&free) {
    free(x_u);
    free(x_v);
    free(y_u);
    free(y_v);
    free(bar_x);
    free(bar_y);
    free(actual_l);
    free(expect_l);
  }
};

template <std::floating_point Real>
__device__ __host__ void GetLength(int i,
    Real const *memory_x_u, Real const *memory_x_v,
    Real const *memory_y_u, Real const *memory_y_v,
    Real *bar_x, Real *bar_y, Real *actual_l, Real *expect_l) {
  Real x_u = memory_x_u[i];
  Real x_v = memory_x_v[i];
  Real y_u = memory_y_u[i];
  Real y_v = memory_y_v[i];
  actual_l[i] = std::hypot(bar_x[i] = x_v - x_u,
                           bar_y[i] = y_v - y_u);
  expect_l[i] = scaling((x_u + x_v) / 2,
                        (y_u + y_v) / 2);
  assert(actual_l[i] >= 0);
  assert(expect_l[i] >= 0);
}

template <std::floating_point Real>
void HostGetLengths(int n_edge, Memory<Real> const &host_memory_pinned,
    Real *bar_x, Real *bar_y, Real *actual_l, Real *expect_l) {
  for (int i = 0; i < n_edge; i++) {
    GetLength(i,
        host_memory_pinned.x_u, host_memory_pinned.x_v,
        host_memory_pinned.y_u, host_memory_pinned.y_v,
        bar_x, bar_y, actual_l, expect_l);
  }
}


template <std::floating_point Real>
__global__ void DeviceGetLength(int n_edge,
    Real const *device_memory_x_u, Real const *device_memory_x_v,
    Real const *device_memory_y_u, Real const *device_memory_y_v,
    Real *bar_x, Real *bar_y, Real *actual_l, Real *expect_l) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_edge) {
    return;
  }
  GetLength(i,
      device_memory_x_u, device_memory_x_v,
      device_memory_y_u, device_memory_y_v,
      bar_x, bar_y, actual_l, expect_l);
}

template <std::floating_point Real>
void DeviceGetLengths(int n_edge, Memory<Real> const &host_memory_pinned,
    Memory<Real> *device_memory,
    Real *bar_x, Real *bar_y, Real *actual_l, Real *expect_l) {
  // Initialize streams for concurrency:
  auto streams = new cudaStream_t[NSTREAMS];
  for (int i = 0; i < NSTREAMS; i++) {
    cudaStreamCreate(&streams[i]);
  } 
  int n_edge_per_stream = n_edge / NSTREAMS;
  int n_byte = n_edge_per_stream * sizeof(Real);
  // Dispatch comm and comp for n_edge_per_stream edges to NSTREAMS streams:
  for (int i_stream = 0; i_stream < NSTREAMS; i_stream++) {
    int offset = i_stream * n_edge_per_stream;
    if (i_stream + 1 == NSTREAMS) {
      n_edge_per_stream = n_edge - i_stream * n_edge_per_stream;
      n_byte = n_edge_per_stream * sizeof(Real);
    }
    // std::cout << "i_stream = " << i_stream << "\n";
    // std::cout << "  n_edge_per_stream = " << n_edge_per_stream << ", ";
    // std::cout << "  n_byte = " << n_byte << "\n";
    dim3 block = std::min(256, n_edge_per_stream);
    dim3 grid = (n_edge_per_stream + block.x - 1) / block.x;
    // std::cout << "  block.x = " << block.x << ", ";
    // std::cout << "  grid.x = " << grid.x << "\n";
    auto stream_i = streams[i_stream];
    // copy input data from host to device
    cudaMemcpyAsync(device_memory->x_u + offset, host_memory_pinned.x_u + offset,
        n_byte, cudaMemcpyHostToDevice, stream_i);
    cudaMemcpyAsync(device_memory->x_v + offset, host_memory_pinned.x_v + offset,
        n_byte, cudaMemcpyHostToDevice, stream_i);
    cudaMemcpyAsync(device_memory->y_u + offset, host_memory_pinned.y_u + offset,
        n_byte, cudaMemcpyHostToDevice, stream_i);
    cudaMemcpyAsync(device_memory->y_v + offset, host_memory_pinned.y_v + offset,
        n_byte, cudaMemcpyHostToDevice, stream_i);
    // execute the kernel
    DeviceGetLength<<< grid, block, 0, stream_i >>>(n_edge_per_stream,
        device_memory->x_u + offset, device_memory->x_v + offset,
        device_memory->y_u + offset, device_memory->y_v + offset,
        device_memory->bar_x + offset, device_memory->bar_y + offset,
        device_memory->actual_l + offset, device_memory->expect_l + offset);
    // copy output data from device to host
    cudaMemcpyAsync(bar_x + offset, device_memory->bar_x + offset,
        n_byte, cudaMemcpyDeviceToHost, stream_i);
    cudaMemcpyAsync(bar_y + offset, device_memory->bar_y + offset,
        n_byte, cudaMemcpyDeviceToHost, stream_i);
    cudaMemcpyAsync(actual_l + offset, device_memory->actual_l + offset,
        n_byte, cudaMemcpyDeviceToHost, stream_i);
    cudaMemcpyAsync(expect_l + offset, device_memory->expect_l + offset,
        n_byte, cudaMemcpyDeviceToHost, stream_i);
  }
  cudaDeviceSynchronize();
  for (int i = 0; i < NSTREAMS; i++) {
    cudaStreamDestroy(streams[i]);
  }
  delete[] streams;
}

template <typename T>
T Get(YAML::Node const &config, char const *key) {
  try {
    return config[key].as<T>();
  } catch (...) {
    std::fprintf(stderr, "Failed to parse the key \"%s\"\n", key);
    throw;
  }
}

int main(int argc, char *argv[]) {
  std::srand(31415926);

  using Real = double;
  using Column = HostDynamicVector<Real>;

  if (argc < 2) {
    std::cout << "usage:\n    ./distance <config.yaml>\n";
    return -1;
  }
  YAML::Node config = YAML::LoadFile(argv[1]);

  int n_point = Get<int>(config, "n_point");
  int n_frame = Get<int>(config, "n_frame");  // maximum writing step
  int n_step_per_frame = Get<int>(config, "n_step_per_frame");
  int n_step = n_step_per_frame * n_frame;  // maximum iteration step

  Real const h_0 = Get<Real>(config, "h_0");
  Real const eps = 1e-16;
  auto g_eps = h_0 * 0.001;  // for rejecting out-of-domain faces
  auto d_eps = h_0 * std::sqrt(eps);  // for finite-differencing d(x, y)

  float elapsed_time = 0.0, host_cost = 0.0, device_cost = 0.0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop); 

  cudaEventRecord(start);  // timing the initialization
  // Build random points in \f$ [-1, 1]^2 \times 0 \f$
  Column x(n_point), y(n_point), z(n_point);
  x.setRandom(); y.setRandom(); z.setZero();
  // Fix corner points:
  x[0] = x[2] = x_min;
  x[1] = x[3] = x_max;
  y[0] = y[1] = y_min;
  y[2] = y[3] = y_max;

  // Reject out-of-domain (d > 0) points:
  n_point = RejectPoints(&x, &y, distance<Real>);
  assert(n_point == x.size());
  assert(n_point == y.size());

  // Pre-allocate memory:
  int n_edge_max = n_point * 4;
  auto host_memory = Memory<Real>();
  host_memory.Malloc(n_edge_max * sizeof(Real),
      [](void **ptr, int n_byte) { *ptr = std::malloc(n_byte); });
  Real *host_bar_x = host_memory.bar_x;
  Real *host_bar_y = host_memory.bar_y;
  Real *host_actual_l = host_memory.actual_l;
  Real *host_expect_l = host_memory.expect_l;

  auto host_memory_pinned = Memory<Real>();
  host_memory_pinned.Malloc(n_edge_max * sizeof(Real),
      [](void **ptr, int n_byte) { cudaHostAlloc(ptr, n_byte,
          cudaHostAllocDefault); });
  Real *bar_x = host_memory_pinned.bar_x;
  Real *bar_y = host_memory_pinned.bar_y;
  Real *actual_l = host_memory_pinned.actual_l;
  Real *expect_l = host_memory_pinned.expect_l;

  auto device_memory = Memory<Real>();
  device_memory.Malloc(n_edge_max * sizeof(Real),
      [](void **ptr, int n_byte) { cudaMalloc(ptr, n_byte); });

  // Triangulate the points.
  using Kernel = CGAL::Simple_cartesian<Real>;
  using Point = Kernel::Point_3;
  using GeoTraits = CGAL::Projection_traits_xy_3<Kernel>;
  using Delaunay = CGAL::Delaunay_triangulation_2<GeoTraits>;

  Real const max_shift_tol = 1.e-3;  // terminate if max_shift / h_0 < this value
  Real max_shift_square = 1.e100;

  std::vector<std::array<int, 3>> faces; int n_face;
  std::vector<std::array<int, 2>> edges; int n_edge;

  auto Triangulate = [&faces, &edges, &x, &y, &z, &n_face, &n_edge, g_eps](int n_point) {
    std::cout << "Re-triangulate.\n";
    auto delaunay = Delaunay();
    for (int i = 0; i < n_point; i++) {
      delaunay.insert(Point(x[i], y[i], z[i]));
    }
    faces = GetFaces(delaunay);
    n_face = RejectFaces(&faces, x, y, distance<Real>, g_eps);
    assert(n_face <= delaunay.number_of_faces());
    edges = GetEdges(faces);
    n_edge = edges.size();
    assert(n_edge <= n_point * 4);
  };

  Real delaunay_tol = 1.e-1;  // re-triangluate if it is < max_shift
  Column x_old = x;
  Column y_old = y;
  // Eigen does not support `Vector += double` ?
  // x_old += delaunay_tol;
  // y_old += delaunay_tol;
  for (int i = 0; i < n_point; ++i) {
    x_old[i] += delaunay_tol * 2;
  }

  auto HasLargeShift = [](Real delaunay_tol,
      Column const &x_old, Column const &x,
      Column const &y_old, Column const &y) {
    Column norms = (
        (x_old - x).array().square() +
        (y_old - y).array().square()).sqrt();
    return norms.maxCoeff() > delaunay_tol;
  };

  int too_close_freq = 25;
  auto too_close = std::unordered_set<int>();

  auto RemoveTooClosePoints = [&too_close, &x, &y](int n)
      -> int {
    if (too_close.empty()) {
      return n;
    }
    int n_new = n - too_close.size();
    Column x_new(n_new), y_new(n_new);
    int i_new = 0;
    for (int i = 0; i < n; i++) {
      if (too_close.find(i) == too_close.end()) {
        x_new[i_new] = x[i];
        y_new[i_new] = y[i];
        i_new++;
      }
    }
    x = x_new;
    y = y_new;
    return n_new;
  };

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  printf("Initialization costs %.2f ms\n", elapsed_time);

  // The main loop:
  for (int i_step = 0; i_step <= n_step; i_step++) {
    n_point = RemoveTooClosePoints(n_point);
    if (too_close.size() || HasLargeShift(delaunay_tol, x_old, x, y_old, y)
        || i_step % n_step_per_frame == 0) {
      too_close.clear();
      x_old = x;
      y_old = y;
      Triangulate(n_point);
    }

    // Write the points and triangles.
    if (i_step % n_step_per_frame == 0) {
      int i_frame = i_step / n_step_per_frame;
      WriteFrame(i_frame, n_point, x.data(), y.data(), z.data(), faces);
    }

    // Collect point data:
    for (int i = 0; i < n_edge; i++) {
      auto [u, v] = edges[i];
      host_memory_pinned.x_u[i] = x[u];
      host_memory_pinned.x_v[i] = x[v];
      host_memory_pinned.y_u[i] = y[u];
      host_memory_pinned.y_v[i] = y[v];
    }

    // Build actual and expect lengths:
    cudaEventRecord(start);
    HostGetLengths(n_edge, host_memory_pinned,
        host_bar_x, host_bar_y, host_actual_l, host_expect_l);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    host_cost += elapsed_time;

    cudaEventRecord(start);
    DeviceGetLengths(n_edge, host_memory_pinned, &device_memory,
        bar_x, bar_y, actual_l, expect_l);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    device_cost += elapsed_time;

    for (int i = 0; i < n_edge; i++) {
      host_bar_x[i] = std::abs(host_bar_x[i] - bar_x[i]);
      host_bar_y[i] = std::abs(host_bar_y[i] - bar_y[i]);
      host_actual_l[i] = std::abs(host_actual_l[i] - actual_l[i]);
      host_expect_l[i] = std::abs(host_expect_l[i] - expect_l[i]);
    }

    std::cout << "|diff_bar_x| + |diff_bar_y| = " <<
        std::accumulate(host_bar_x, host_bar_x + n_edge, 0.) +
        std::accumulate(host_bar_y, host_bar_y + n_edge, 0.) << "\n";
    std::cout << "|diff_actual_l| + |diff_expect_l| = " <<
        std::accumulate(host_actual_l, host_actual_l + n_edge, 0.) +
        std::accumulate(host_expect_l, host_expect_l + n_edge, 0.) << "\n";

    auto actual_norm = std::inner_product(actual_l, actual_l + n_edge, actual_l, 0.);
    auto expect_norm = std::inner_product(expect_l, expect_l + n_edge, expect_l, 0.);
    Real norm_ratio = 1.2 * std::sqrt(actual_norm / expect_norm);
    std::cout << actual_norm << " " << expect_norm << " " << norm_ratio << "\n";
    std::for_each_n(expect_l, n_edge,
        [norm_ratio](Real &x) { x *= norm_ratio; });

    // Label too-close points:
    if ((i_step + 1) % too_close_freq == 0) {
      assert(too_close.empty());
      for (int i = 0; i < n_edge; ++i) {
        if (expect_l[i] > 2 * actual_l[i]) {
          auto [u, v] = edges[i];
          if (u >= n_fixed) {
            too_close.emplace(u);
          }
          if (v >= n_fixed) {
            too_close.emplace(v);
          }
        }
      }
      std::cout << too_close.size() << " points should be removed.\n";
      continue;
    }

    // Get forces at nodes:
    Column force_x(n_point), force_y(n_point);
    force_x.setZero(); force_y.setZero();
    for (int i = 0; i < n_edge; i++) {
      // repulsive force for compressed bars
      Real force = /* 1.0 * */std::max(0., expect_l[i] / actual_l[i] - 1);
      auto [u, v] = edges[i];
      force_x[u] -= force * bar_x[i];
      force_x[v] += force * bar_x[i];
      force_y[u] -= force * bar_y[i];
      force_y[v] += force * bar_y[i];
    }
    // No force at fixed points:
    for (int i = 0; i < n_fixed; i++) {
      force_x[i] = force_y[i] = 0.;
    }

    // Move points:
    Real delta_t = 0.2;
    Column shift_x(n_point), shift_y(n_point);
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
      auto fix_x = grad_x * d;
      auto fix_y = grad_y * d;
      x[i] -= fix_x;
      y[i] -= fix_y;
      shift_x[i] -= fix_x;
      shift_y[i] -= fix_y;
    }

    // Update the maximum shift:
    max_shift_square = 0.;
    for (int i = n_fixed; i < n_point; i++) {
      if (distance(x[i], y[i]) > -g_eps) {
        continue;
      }
      max_shift_square = std::max(max_shift_square,
          shift_x[i] * shift_x[i] + shift_y[i] * shift_y[i]);
    }
    auto max_shift = std::sqrt(max_shift_square) / h_0;

    std::cout << "Step " << i_step << ", n_edge = " << n_edge << ", max_shift = " << max_shift << "\n";

    if (max_shift < max_shift_tol) {
      int i_frame = (i_step + n_step_per_frame - 1) / n_step_per_frame;
      WriteFrame(i_frame, n_point, x.data(), y.data(), z.data(), faces);
      std::cout << "Converged at Step " << i_step << ", Frame " << i_frame << ".\n";
      break;
    }
  }  // main loop

  host_memory.Free([](void *ptr) { std::free(ptr); });
  host_memory_pinned.Free([](void *ptr) { cudaFreeHost(ptr); });
  device_memory.Free([](void *ptr) { cudaFree(ptr); });

  printf("The host costs %.2f ms\n", host_cost);
  printf("The device costs %.2f ms\n", device_cost);

  return 0;
}
