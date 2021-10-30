//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "mini/data/path.hpp"  // defines TEST_DATA_DIR
#include "mini/mesh/mapper/cgns_to_metis.hpp"
#include "mini/mesh/cgns/format.hpp"
#include "mini/mesh/cgns/part.hpp"
#include "mini/mesh/metis/format.hpp"
#include "mini/mesh/metis/partitioner.hpp"
#include "mini/integrator/hexa.hpp"
#include "mini/polynomial/projection.hpp"
#include "mini/polynomial/limiter.hpp"
#include "mini/riemann/euler/eigen.hpp"

#include "gtest/gtest.h"

class TestWenoLimiters : public ::testing::Test {
 protected:
  using Basis = mini::polynomial::OrthoNormal<double, 3, 2>;
  using Gauss = mini::integrator::Hexa<double, 4, 4, 4>;
  using Coord = typename Gauss::GlobalCoord;
  Gauss gauss_;
  std::string const test_data_dir_{TEST_DATA_DIR};
};
TEST_F(TestWenoLimiters, ReconstructScalar) {
  auto case_name = std::string("simple_cube");
  // build mesh files
  constexpr int kCommandLength = 1024;
  char cmd[kCommandLength];
  std::snprintf(cmd, kCommandLength, "mkdir -p %s/scalar", case_name.c_str());
  std::system(cmd); std::cout << "[Done] " << cmd << std::endl;
  auto old_file_name = case_name + "/scalar/original.cgns";
  std::snprintf(cmd, kCommandLength, "gmsh %s/%s.geo -save -o %s",
      test_data_dir_.c_str(), case_name.c_str(), old_file_name.c_str());
  std::system(cmd); std::cout << "[Done] " << cmd << std::endl;
  using CgnsMesh = mini::mesh::cgns::File<double>;
  auto cgns_mesh = CgnsMesh(old_file_name);
  cgns_mesh.ReadBases();
  using Mapper = mini::mesh::mapper::CgnsToMetis<double, idx_t>;
  auto mapper = Mapper();
  auto metis_mesh = mapper.Map(cgns_mesh);
  EXPECT_TRUE(mapper.IsValid());
  // get adjacency between cells
  idx_t n_common_nodes{3};
  auto graph = mini::mesh::metis::MeshToDual(metis_mesh, n_common_nodes);
  int n_cells = metis_mesh.CountCells();
  auto cell_adjs = std::vector<std::vector<int>>(n_cells);
  for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
    for (int r = graph.range(i_cell); r < graph.range(i_cell+1); ++r) {
      int j_cell = graph.index(r);
      cell_adjs[i_cell].emplace_back(j_cell);
    }
  }
  // build cells and project the function on them
  using Cell = mini::mesh::cgns::Cell<int, double, 1>;
  auto cells = std::vector<Cell>();
  cells.reserve(n_cells);
  auto& zone = cgns_mesh.GetBase(1).GetZone(1);
  auto& coordinates = zone.GetCoordinates();
  auto& x = coordinates.x();
  auto& y = coordinates.y();
  auto& z = coordinates.z();
  auto& sect = zone.GetSection(1);
  auto func = [](Coord const &xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    return (x-1.5)*(x-1.5) + (y-1.5)*(y-1.5) + 10*(x < y ? 2. : 0.);
  };
  using Mat3x8 = mini::algebra::Matrix<double, 3, 8>;
  for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
    Mat3x8 coords;
    const cgsize_t* array;  // head of 1-based-node-id list
    array = sect.GetNodeIdListByOneBasedCellId(i_cell+1);
    for (int i = 0; i < 8; ++i) {
      auto i_node = array[i] - 1;
      coords(0, i) = x[i_node];
      coords(1, i) = y[i_node];
      coords(2, i) = z[i_node];
    }
    auto hexa_ptr = std::make_unique<Gauss>(coords);
    cells.emplace_back(std::move(hexa_ptr), i_cell);
    assert(&(cells[i_cell]) == &(cells.back()));
    cells[i_cell].Project(func);
  }
  using Projection = typename Cell::Projection;
  auto adj_projections = std::vector<std::vector<Projection>>(n_cells);
  using Mat1x1 = mini::algebra::Matrix<double, 1, 1>;
  auto adj_smoothness = std::vector<std::vector<Mat1x1>>(n_cells);
  for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
    auto& cell_i = cells[i_cell];
    adj_smoothness[i_cell].emplace_back(cell_i.func_.GetSmoothness());
    for (auto j_cell : cell_adjs[i_cell]) {
      auto adj_func = [&](Coord const &xyz) {
        return cells[j_cell].func_(xyz);
      };
      adj_projections[i_cell].emplace_back(adj_func, cell_i.basis_);
      auto& adj_projection = adj_projections[i_cell].back();
      Mat1x1 diff = cell_i.func_.GetAverage() - adj_projection.GetAverage();
      adj_projection += diff;
      diff = cell_i.func_.GetAverage() - adj_projection.GetAverage();
      EXPECT_NEAR(diff.cwiseAbs().maxCoeff(), 0.0, 1e-14);
      adj_smoothness[i_cell].emplace_back(adj_projection.GetSmoothness());
    }
  }
  const double eps = 1e-6, w0 = 0.001;
  for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
    int adj_cnt = cell_adjs[i_cell].size();
    auto weights = std::vector<double>(adj_cnt + 1, w0);
    weights.back() = 1 - w0 * adj_cnt;
    for (int i = 0; i <= adj_cnt; ++i) {
      auto temp = eps + adj_smoothness[i_cell][i][0];
      weights[i] /= temp * temp;
    }
    auto sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (int j_cell = 0; j_cell <= adj_cnt; ++j_cell) {
      weights[j_cell] /= sum;
    }
    auto& projection_i = cells[i_cell].func_;
    projection_i *= weights.back();
    for (int j_cell = 0; j_cell < adj_cnt; ++j_cell) {
      projection_i += adj_projections[i_cell][j_cell] *= weights[j_cell];
    }
    std::printf("%8.2f (%2d) <- {%8.2f",
        projection_i.GetSmoothness()[0], i_cell,
        adj_smoothness[i_cell].back()[0]);
    for (int j_cell = 0; j_cell < adj_cnt; ++j_cell)
      std::printf(" %8.2f (%2d <- %-2d)", adj_smoothness[i_cell][j_cell][0],
          i_cell, cell_adjs[i_cell][j_cell]);
    std::printf(" }\n");
  }
}
TEST_F(TestWenoLimiters, For3dEulerEquations) {
  auto case_name = std::string("simple_cube");
  // build a cgns mesh file
  constexpr int kCommandLength = 1024;
  char cmd[kCommandLength];
  std::snprintf(cmd, kCommandLength, "mkdir -p %s/vector", case_name.c_str());
  std::system(cmd); std::cout << "[Done] " << cmd << std::endl;
  auto old_file_name = case_name + "/vector/original.cgns";
  std::snprintf(cmd, kCommandLength, "gmsh %s/%s.geo -save -o %s",
      test_data_dir_.c_str(), case_name.c_str(), old_file_name.c_str());
  std::system(cmd); std::cout << "[Done] " << cmd << std::endl;
  using CgnsMesh = mini::mesh::cgns::File<double>;
  auto cgns_mesh = CgnsMesh(old_file_name);
  cgns_mesh.ReadBases();
  // build the dual graph
  idx_t n_common_nodes{3};
  auto mapper = mini::mesh::mapper::CgnsToMetis<double, idx_t>();
  auto metis_mesh = mapper.Map(cgns_mesh);
  EXPECT_TRUE(mapper.IsValid());
  auto graph = mini::mesh::metis::MeshToDual(metis_mesh, n_common_nodes);
  int n_cells = metis_mesh.CountCells();
  // build cells and project a vector function on them
  using Cell = mini::mesh::cgns::Cell<int, double, 5>;
  auto cells = std::vector<Cell>();
  cells.reserve(n_cells);
  auto& zone = cgns_mesh.GetBase(1).GetZone(1);
  auto& coordinates = zone.GetCoordinates();
  auto& x = coordinates.x();
  auto& y = coordinates.y();
  auto& z = coordinates.z();
  auto& sect = zone.GetSection(1);
  // define the function
  using Mat5x1 = mini::algebra::Matrix<double, 5, 1>;
  auto func = [](Coord const &xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    Mat5x1 res;
    res[0] = x * x + 10 * (x < y ? +1 : 0.125);
    res[1] =          2 * (x < y ? -2 : 2);
    res[2] =          2 * (x < y ? -2 : 2);
    res[3] =          2 * (x < y ? -2 : 2);
    res[4] = y * y + 90 * (x < y ? +1 : 0.5);
    return res;
  };
  using Mat3x8 = mini::algebra::Matrix<double, 3, 8>;
  for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
    // build a new `Cell`
    Mat3x8 coords;
    const cgsize_t* array;  // head of 1-based-node-id list
    array = sect.GetNodeIdListByOneBasedCellId(i_cell+1);
    for (int i = 0; i < 8; ++i) {
      auto i_node = array[i] - 1;
      coords(0, i) = x[i_node];
      coords(1, i) = y[i_node];
      coords(2, i) = z[i_node];
    }
    auto hexa_ptr = std::make_unique<Gauss>(coords);
    cells.emplace_back(std::move(hexa_ptr), i_cell);
    assert(&(cells[i_cell]) == &(cells.back()));
    // project `func` onto the latest built cell
    cells[i_cell].Project(func);
  }
  // fill `adj_cells_`
  for (int i_cell = 0; i_cell < n_cells; ++i_cell) {
    for (int r = graph.range(i_cell); r < graph.range(i_cell+1); ++r) {
      int j_cell = graph.index(r);
      cells[i_cell].adj_cells_.emplace_back(&cells[j_cell]);
    }
  }
  // reconstruct using a `EigenWeno` limiter
  using Projection = typename Cell::Projection;
  using IdealGas = mini::riemann::euler::IdealGas<1, 4>;
  using Matrices = mini::riemann::euler::EigenMatrices<double, IdealGas>;
  auto eigen_limiter = mini::polynomial::EigenWeno<Cell, Matrices>(
      /* w0 = */0.01, /* eps = */1e-6);
  auto lazy_limiter = mini::polynomial::LazyWeno<Cell>(
      /* w0 = */0.01, /* eps = */1e-6, /* verbose = */true);
  auto eigen_projections = std::vector<Projection>();
  eigen_projections.reserve(n_cells);
  auto lazy_projections = std::vector<Projection>();
  lazy_projections.reserve(n_cells);
  for (auto& cell : cells) {
    //  lasy limiter
    lazy_projections.emplace_back(lazy_limiter(cell));
    auto lazy_smoothness = lazy_projections.back().GetSmoothness();
    std::printf("\n lazy smoothness[%2d] = ", cell.metis_id);
    std::cout << std::scientific << std::setprecision(3)
        << lazy_smoothness.transpose();
    // eigen limiter
    eigen_projections.emplace_back(eigen_limiter(cell));
    auto eigen_smoothness = eigen_projections.back().GetSmoothness();
    std::printf("\neigen smoothness[%2d] = ", cell.metis_id);
    std::cout << std::scientific << std::setprecision(3)
        << eigen_smoothness.transpose();
    std::cout << std::endl;
    Mat5x1 diff = cell.func_.GetAverage()
        - eigen_projections.back().GetAverage();
    EXPECT_NEAR(diff.cwiseAbs().maxCoeff(), 0.0, 1e-13);
    diff = cell.func_.GetAverage() - lazy_projections.back().GetAverage();
    EXPECT_NEAR(diff.cwiseAbs().maxCoeff(), 0.0, 1e-13);
  }
  std::cout << std::endl;
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
