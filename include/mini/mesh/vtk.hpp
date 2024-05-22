// Copyright 2023 PEI Weicheng
#ifndef MINI_MESH_VTK_HPP_
#define MINI_MESH_VTK_HPP_

#include <bit>
#include <fstream>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace mini {
namespace mesh {
namespace vtk {

using Byte = char;

std::string EncodeBase64(Byte const *input_data, std::size_t n_char) {
  auto output = std::string();
  output.resize((n_char / 3 + (n_char % 3 != 0)) * 4);
  auto *output_data = output.data();
  auto *output_end = output_data + output.size();
  auto *input_end = input_data + n_char;
  constexpr char base[]
      = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  auto encode = [&base](Byte *out, Byte const *in) {
    out[0] = base[((in[0] & 0xFC/* 1111 1100 */) >> 2)];
    out[1] = base[((in[0] & 0x03/* 0000 0011 */) << 4)
                + ((in[1] & 0xF0/* 1111 0000 */) >> 4)];
    out[2] = base[((in[1] & 0x0F/* 0000 1111 */) << 2)
                + ((in[2] & 0xC0/* 1100 0000 */) >> 6)];
    out[3] = base[((in[2] & 0x3F/* 0011 1111 */))];
  };
  for (std::size_t i_triplet = 0, n_triplet = n_char / 3;
      i_triplet < n_triplet; i_triplet++) {
    // for each (char[3]) input_data, convert it to (char[4]) output_data:
    encode(output_data, input_data);
    input_data += 3;
    output_data += 4;
  }
  assert(input_data <= input_end);
  if (input_data != input_end) {  // need padding
    Byte triplet[3] = { 0, 0, 0 };
    int i = 0;
    while (input_data != input_end) {
      triplet[i++] = *input_data++;
    }
    assert(i == 1 || i == 2);
    i++;  // 1 in input -> 2 in output, 2 in input -> 3 in output
    encode(output_data, triplet);
    output_data += i;
    while (output_data != output_end) {
      *output_data++ = '=';
      i++;
    }
    assert(i == 4);
  }
  return output;
}

template <typename Iter>
std::pair<std::string, std::size_t> EncodeBase64(Iter first, Iter last) {
  auto *input_data = reinterpret_cast<Byte const *>(&first[0]);
  std::size_t n_char = (last - first) * sizeof(decltype(*first));
  return {EncodeBase64(input_data, n_char), n_char};
}

/**
 * @brief Mimic VTK's cell types.
 * 
 * See [vtkCellType.h](https://vtk.org/doc/nightly/html/vtkCellType_8h.html) for details.
  */
enum class CellType {
  kTetrahedron4 = 10,
  kHexahedron8 = 12,
  kWedge6 = 13,
  kWedge15 = 26,
  kTetrahedron10 = 24,
  kHexahedron20 = 25,
  kHexahedron27 = 29,
  kHexahedron64 = 72,
};

/**
 * @brief Mimic VTK's cells.
 * 
 * For node numbering, see [linear element](https://raw.githubusercontent.com/Kitware/vtk-examples/gh-pages/src/VTKBook/Figures/Figure5-2.png), [high-order element](https://raw.githubusercontent.com/Kitware/vtk-examples/gh-pages/src/VTKBook/Figures/Figure5-4.png) and [arbitrary Lagrange element](https://gitlab.kitware.com/vtk/vtk/uploads/d18be24480da192e4b70568f050d114f/VtkLagrangeNodeNumbering.pdf) for details.
 */
class Element {
};

/**
 * @brief Mimic VTK's [vtkTetra](https://vtk.org/doc/nightly/html/classvtkTetra.html).
 * 
 * @tparam Local  Type of local coordinates.
 */
template <typename Local>
class Tetrahedron4 : public Element {
 public:
  static const Local locals[4];
};
template <typename Local>
const Local Tetrahedron4<Local>::locals[4]{
  Local(1, 0, 0), Local(0, 1, 0), Local(0, 0, 1), Local(0, 0, 0)
};

/**
 * @brief Mimic VTK's [vtkQuadraticTetra](https://vtk.org/doc/nightly/html/classvtkQuadraticTetra.html).
 * 
 * @tparam Local  Type of local coordinates.
 */
template <typename Local>
class Tetrahedron10 : public Element {
 public:
  static const Local locals[10];
};
template <typename Local>
const Local Tetrahedron10<Local>::locals[10]{
  Local(1, 0, 0), Local(0, 1, 0), Local(0, 0, 1), Local(0, 0, 0),
  Local(0.5, 0.5, 0), Local(0, 0.5, 0.5), Local(0.5, 0, 0.5),
  Local(0.5, 0, 0), Local(0, 0.5, 0), Local(0, 0, 0.5),
};

/**
 * @brief Mimic VTK's [vtkWedge](https://vtk.org/doc/nightly/html/classvtkWedge.html).
 * 
 * @tparam Local  Type of local coordinates.
 */
template <typename Local>
class Wedge6 : public Element {
 public:
  static const Local locals[6];
};
template <typename Local>
const Local Wedge6<Local>::locals[6]{
  /**
   *   VTK's 0, 1, 2, 3, 4, 5 correspond to
   *  CGNS's 1, 3, 2, 4, 6, 5
   */
  Local(1, 0, -1), Local(0, 0, -1), Local(0, 1, -1),
  Local(1, 0, +1), Local(0, 0, +1), Local(0, 1, +1),
};

/**
 * @brief Mimic VTK's [vtkQuadraticWedge](https://vtk.org/doc/nightly/html/classvtkQuadraticWedge.html).
 * 
 * @tparam Local  Type of local coordinates.
 */
template <typename Local>
class Wedge15 : public Element {
 public:
  static const Local locals[15];
};
template <typename Local>
const Local Wedge15<Local>::locals[15]{
  // nodes at bottom (0, 1, 2) and top (3, 4, 5) corners
  Local(1, 0, -1), Local(0, 1, -1), Local(0, 0, -1),
  Local(1, 0, +1), Local(0, 1, +1), Local(0, 0, +1),
  // nodes on bottom (6, 7, 8) and top (9, 10, 11) edges
  Local(0.5, 0.5, -1), Local(0, 0.5, -1), Local(0.5, 0, -1),
  Local(0.5, 0.5, +1), Local(0, 0.5, +1), Local(0.5, 0, +1),
  // nodes on vertical (12, 13, 14) edges
  Local(1, 0, 0), Local(0, 1, 0), Local(0, 0, 0),
};

/**
 * @brief Mimic VTK's [vtkHexahedron](https://vtk.org/doc/nightly/html/classvtkHexahedron.html).
 * 
 * @tparam Local  Type of local coordinates.
 */
template <typename Local>
class Hexahedron8 : public Element {
 public:
  static const Local locals[8];
};
template <typename Local>
const Local Hexahedron8<Local>::locals[8]{
  Local(-1, -1, -1), Local(+1, -1, -1), Local(+1, +1, -1), Local(-1, +1, -1),
  Local(-1, -1, +1), Local(+1, -1, +1), Local(+1, +1, +1), Local(-1, +1, +1),
};

/**
 * @brief Mimic VTK's [vtkQuadraticHexahedron](https://vtk.org/doc/nightly/html/classvtkQuadraticHexahedron.html).
 * 
 * @tparam Local  Type of local coordinates.
 */
template <typename Local>
class Hexahedron20 : public Element {
 public:
  static const Local locals[20];
};
template <typename Local>
const Local Hexahedron20<Local>::locals[20]{
  // nodes at corners (same as Hexahedron8)
  Local(-1, -1, -1), Local(+1, -1, -1), Local(+1, +1, -1), Local(-1, +1, -1),
  Local(-1, -1, +1), Local(+1, -1, +1), Local(+1, +1, +1), Local(-1, +1, +1),
  // nodes on edges
  Local(0., -1, -1), Local(+1, 0., -1), Local(0., +1, -1), Local(-1, 0., -1),
  Local(0., -1, +1), Local(+1, 0., +1), Local(0., +1, +1), Local(-1, 0., +1),
  Local(-1, -1, 0.), Local(+1, -1, 0.), Local(+1, +1, 0.), Local(-1, +1, 0.),
};

/**
 * @brief Mimic VTK's [vtkTriQuadraticHexahedron](https://vtk.org/doc/nightly/html/classvtkTriQuadraticHexahedron.html).
 * 
 * @tparam Local  Type of local coordinates.
 */
template <typename Local>
class Hexahedron27 : public Element {
 public:
  static const Local locals[27];
};
template <typename Local>
const Local Hexahedron27<Local>::locals[27]{
  // nodes at corners (same as Hexahedron8)
  Local(-1, -1, -1), Local(+1, -1, -1), Local(+1, +1, -1), Local(-1, +1, -1),
  Local(-1, -1, +1), Local(+1, -1, +1), Local(+1, +1, +1), Local(-1, +1, +1),
  // nodes on edges
  Local(0., -1, -1), Local(+1, 0., -1), Local(0., +1, -1), Local(-1, 0., -1),
  Local(0., -1, +1), Local(+1, 0., +1), Local(0., +1, +1), Local(-1, 0., +1),
  Local(-1, -1, 0.), Local(+1, -1, 0.), Local(+1, +1, 0.), Local(-1, +1, 0.),
  // nodes on faces
  Local(-1, 0., 0.), Local(+1, 0., 0.),
  Local(0., -1, 0.), Local(0., +1, 0.),
  Local(0., 0., -1), Local(0., 0., +1),
  // node on center
  Local(0., 0., 0.),
};

/**
 * @brief Mimic VTK's [vtkLagrangeHexahedron](https://vtk.org/doc/nightly/html/classvtkLagrangeHexahedron.html).
 * 
 * @tparam Local  Type of local coordinates.
 */
template <typename Local>
class Hexahedron64 : public Element {
  static constexpr double a = 1. / 3;

 public:
  static const Local locals[64];
};
template <typename Local>
const Local Hexahedron64<Local>::locals[64]{
  // nodes at corners (same as Hexahedron8)
  Local(-1, -1, -1), Local(+1, -1, -1), Local(+1, +1, -1), Local(-1, +1, -1),
  Local(-1, -1, +1), Local(+1, -1, +1), Local(+1, +1, +1), Local(-1, +1, +1),
  // [8, 16) nodes on bottom edges
  Local(-a, -1, -1), Local(+a, -1, -1), Local(+1, -a, -1), Local(+1, +a, -1),
  Local(-a, +1, -1), Local(+a, +1, -1), Local(-1, -a, -1), Local(-1, +a, -1),
  // [16, 24) nodes on top edges
  Local(-a, -1, +1), Local(+a, -1, +1), Local(+1, -a, +1), Local(+1, +a, +1),
  Local(-a, +1, +1), Local(+a, +1, +1), Local(-1, -a, +1), Local(-1, +a, +1),
  // [24, 32) nodes on vertical edges
  Local(-1, -1, -a), Local(-1, -1, +a), Local(+1, -1, -a), Local(+1, -1, +a),
  Local(-1, +1, -a), Local(-1, +1, +a), Local(+1, +1, -a), Local(+1, +1, +a),
  // [32, 36) nodes on the left face
  Local(-1, -a, -a), Local(-1, +a, -a), Local(-1, -a, +a), Local(-1, +a, +a),
  // [36, 40) nodes on the right face
  Local(+1, -a, -a), Local(+1, +a, -a), Local(+1, -a, +a), Local(+1, +a, +a),
  // [40, 44) nodes on the front face
  Local(-a, -1, -a), Local(+a, -1, -a), Local(-a, -1, +a), Local(+a, -1, +a),
  // [44, 48) nodes on the back face
  Local(-a, +1, -a), Local(+a, +1, -a), Local(-a, +1, +a), Local(+a, +1, +a),
  // [48, 52) nodes on the bottom face
  Local(-a, -a, -1), Local(+a, -a, -1), Local(-a, +a, -1), Local(+a, +a, -1),
  // [52, 56) nodes on the top face
  Local(-a, -a, +1), Local(+a, -a, +1), Local(-a, +a, +1), Local(+a, +a, +1),
  // [56, 64) nodes inside the body
  Local(-a, -a, -a), Local(+a, -a, -a), Local(-a, +a, -a), Local(+a, +a, -a),
  Local(-a, -a, +a), Local(+a, -a, +a), Local(-a, +a, +a), Local(+a, +a, +a),
};

template <typename Part>
class Writer {
  using Cell = typename Part::Cell;
  using Value = typename Cell::Value;
  using Local = typename Cell::Local;
  using Coord = typename Cell::Global;
  using Scalar = typename Cell::Scalar;
  using Function = std::function<Scalar(Cell const &, Coord const &, Value const &)>;
  using ExtraField = std::pair<std::string, Function>;

  static CellType GetCellType(int n_corners) {
    CellType cell_type;
    switch (n_corners) {
      case 4:
        cell_type = CellType::kTetrahedron10;
        break;
      case 6:
        cell_type = CellType::kWedge15;
        break;
      case 8:
        cell_type = CellType::kHexahedron27;
        break;
      default:
        assert(false);
        break;
    }
    return cell_type;
  }
  static int CountNodes(CellType cell_type) {
    int n_nodes;
    switch (cell_type) {
      case CellType::kTetrahedron4:
        n_nodes = 4;
        break;
      case CellType::kTetrahedron10:
        n_nodes = 10;
        break;
      case CellType::kWedge6:
        n_nodes = 6;
        break;
      case CellType::kWedge15:
        n_nodes = 15;
        break;
      case CellType::kHexahedron8:
        n_nodes = 8;
        break;
      case CellType::kHexahedron20:
        n_nodes = 20;
        break;
      case CellType::kHexahedron27:
        n_nodes = 27;
        break;
      case CellType::kHexahedron64:
        n_nodes = 64;
        break;
      default:
        assert(false);
        break;
    }
    return n_nodes;
  }

  /**
   * @brief Append the data carried by a given Cell to the vectors to be written.
   * 
   * @param cell 
   * @param types 
   * @param coords 
   * @param values 
   */
  static void Prepare(const Cell &cell, std::vector<CellType> *types,
      std::vector<Coord> *coords, std::vector<Value> *values) {
    auto type = GetCellType(cell.CountCorners());
    types->push_back(type);
    // TODO(PVC): dispatch by virtual functions?
    Local const *locals;
    int n;
    switch (type) {
    case CellType::kTetrahedron4:
      locals = &Tetrahedron4<Coord>::locals[0];
      n = 4;
      break;
    case CellType::kTetrahedron10:
      locals = &Tetrahedron10<Coord>::locals[0];
      n = 10;
      break;
    case CellType::kWedge6:
      locals = &Wedge6<Coord>::locals[0];
      n = 6;
      break;
    case CellType::kWedge15:
      locals = &Wedge15<Coord>::locals[0];
      n = 15;
      break;
    case CellType::kHexahedron8:
      locals = &Hexahedron8<Coord>::locals[0];
      n = 8;
      break;
    case CellType::kHexahedron20:
      locals = &Hexahedron20<Coord>::locals[0];
      n = 20;
      break;
    case CellType::kHexahedron27:
      locals = &Hexahedron27<Coord>::locals[0];
      n = 27;
      break;
    case CellType::kHexahedron64:
      locals = &Hexahedron64<Coord>::locals[0];
      n = 64;
      break;
    default:
      assert(false);
      break;
    }
    for (int i = 0; i < n; ++i) {
      auto &global = coords->emplace_back();
      auto &value = values->emplace_back();
      cell.polynomial().LocalToGlobalAndValue(locals[i], &global, &value);
    }
  }

  static std::vector<ExtraField> extra_fields_;

 public:
  static bool LittleEndian() {
    return std::endian::native == std::endian::little;
  }

  /**
   * @brief Add an extra field other than the conservative variables carried by Part.
   * 
   * @tparam F 
   * @param name 
   * @param f 
   */
  template <class F>
  static void AddExtraField(std::string const &name, F &&f) {
    extra_fields_.emplace_back(name, std::forward<F>(f));
  }

  /**
   * @brief Write the solution carried by a given Part to a pvtu file with a given name.
   * 
   * @param part 
   * @param soln_name 
   */
  static void WriteSolutions(const Part &part, std::string const &soln_name) {
    std::string endianness
        = LittleEndian() ? "\"LittleEndian\"" : "\"BigEndian\"";
    // prepare data to be written
    auto types = std::vector<CellType>();
    auto coords = std::vector<Coord>();
    auto values = std::vector<Value>();
    for (const Cell &cell : part.GetLocalCells()) {
      Prepare(cell, &types, &coords, &values);
    }
    // create the pvtu file (which refers to vtu files created by rank[0] and other ranks) by rank[0]
    if (part.mpi_rank() == 0) {
      char temp[1024];
      std::snprintf(temp, sizeof(temp), "%s/%s.pvtu",
          part.GetDirectoryName().c_str(), soln_name.c_str());
      auto pvtu = std::ofstream(temp, std::ios::out);
      pvtu << "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" "
          << "byte_order=" << endianness << " header_type=\"UInt64\">\n";
      pvtu << "  <PUnstructuredGrid GhostLevel=\"1\">\n";
      pvtu << "    <PPointData>\n";
      for (int k = 0; k < Part::kComponents; ++k) {
        pvtu << "      <PDataArray type=\"Float64\" Name=\""
            << part.GetFieldName(k) << "\"/>\n";
      }
      for (auto &[name, func] : extra_fields_) {
        pvtu << "      <PDataArray type=\"Float64\" Name=\""
            << name << "\"/>\n";
      }
      pvtu << "    </PPointData>\n";
      pvtu << "    <PPoints>\n";
      pvtu << "      <PDataArray type=\"Float64\" Name=\"Points\" "
          << "NumberOfComponents=\"3\"/>\n";
      pvtu << "    </PPoints>\n";
      for (int i_part = 0; i_part < part.mpi_size(); ++i_part) {
        pvtu << "    <Piece Source=\"./" << soln_name << '/'
            << i_part << ".vtu\"/>\n";
      }
      pvtu << "  </PUnstructuredGrid>\n";
      pvtu << "</VTKFile>\n";
    }
    // create the vtu file by each rank
    bool binary = false;
    auto format = binary ? "\"binary\"" : "\"ascii\"";
    auto vtu = part.GetFileStream(soln_name, binary, "vtu");
    vtu << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\""
        << " byte_order=" << endianness << " header_type=\"UInt64\">\n";
    vtu << "  <UnstructuredGrid>\n";
    vtu << "    <Piece NumberOfPoints=\"" << coords.size()
        << "\" NumberOfCells=\"" << types.size() << "\">\n";
    vtu << "      <PointData>\n";
    int K = values[0].size();
    for (int k = 0; k < K; ++k) {
      vtu << "        <DataArray type=\"Float64\" Name=\""
          << part.GetFieldName(k) << "\" format=" << format << ">\n";
      for (auto &f : values) {
        vtu << f[k] << ' ';
      }
      vtu << "\n        </DataArray>\n";
    }
    vtu << "      </PointData>\n";
    vtu << "      <DataOnCells>\n";
    vtu << "      </DataOnCells>\n";
    vtu << "      <Points>\n";
    vtu << "        <DataArray type=\"Float64\" Name=\"Points\" "
        << "NumberOfComponents=\"3\" format=" << "\"binary\"" << ">\n";
    // for (auto &xyz : coords) {
    //   for (auto v : xyz) {
    //     vtu << v << ' ';
    //   }
    // }
    static_assert(sizeof(Coord) == sizeof(Scalar) * 3);
    {
      auto [encoded, n_char] = EncodeBase64(coords.begin(), coords.end());
      auto [n_char_encoded, _] = EncodeBase64(&n_char, (&n_char) + 1);
      while (n_char_encoded.size() < 8) {
        n_char_encoded.push_back('=');
      }
      vtu << n_char_encoded;
      vtu << encoded;
    }
    vtu << "\n        </DataArray>\n";
    vtu << "      </Points>\n";
    vtu << "      <Cells>\n";
    vtu << "        <DataArray type=\"Int32\" Name=\"connectivity\" "
        << "format=\"ascii\">\n";
    for (int i_node = 0; i_node < coords.size(); ++i_node) {
      vtu << i_node << ' ';
    }
    vtu << "\n        </DataArray>\n";
    vtu << "        <DataArray type=\"Int32\" Name=\"offsets\" "
        << "format=" << format << ">\n";
    int offset = 0;
    for (auto type : types) {
      offset += CountNodes(type);
      vtu << offset << ' ';
    }
    vtu << "\n        </DataArray>\n";
    vtu << "        <DataArray type=\"UInt8\" Name=\"types\" "
        << "format=" << format << ">\n";
    for (auto type : types) {
      vtu << static_cast<int>(type) << ' ';
    }
    vtu << "\n        </DataArray>\n";
    vtu << "      </Cells>\n";
    vtu << "    </Piece>\n";
    vtu << "  </UnstructuredGrid>\n";
    vtu << "</VTKFile>\n";
  }
};

template <typename Part>
std::vector<typename Writer<Part>::ExtraField>
Writer<Part>::extra_fields_;

}  // namespace vtk
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_VTK_HPP_
