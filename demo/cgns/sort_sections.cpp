// Copyright 2024 PEI Weicheng

#include <cstdio>
#include <algorithm>
#include <chrono>
#include <limits>
#include <string>
#include <vector>

#include "cgnslib.h"
#include "mini/mesh/cgns.hpp"

using File = mini::mesh::cgns::File<double>;
using Zone = mini::mesh::cgns::Zone<double>;

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::printf("usage:\n  %s <cgns_file> <higher_dim_first>\n", argv[0]);
    return -1;
  }
  auto cgns_file = std::string(argv[1]);
  bool higher_dim_first = std::atoi(argv[2]);
  auto file = File(cgns_file);
  file.ReadBases();
  auto &base = file.GetUniqueBase();
  auto &zone = base.GetUniqueZone();

  std::cout << "\nbefore sorting:\n";
  for (int i = 1, n = zone.CountSections(); i <= n; i++) {
    auto const &section = zone.GetSection(i);
    std::cout << section.name()
        << " [" << section.CellIdMin() << ", " << section.CellIdMax() << "]\n";
  }

  zone.SortSectionsByDim(higher_dim_first);

  std::cout << "\nafter sorting:\n";
  for (int i = 1, n = zone.CountSections(); i <= n; i++) {
    auto const &section = zone.GetSection(i);
    std::cout << section.name()
        << " [" << section.CellIdMin() << ", " << section.CellIdMax() << "]\n";
  }

  // write the augmented file
  for (int i = 0; i < 5; ++i) {
    cgns_file.pop_back();
  }
  auto output = cgns_file + "_sorted.cgns";
  std::printf("writing to %s ...\n", output.c_str());
  file.Write(output);
}
