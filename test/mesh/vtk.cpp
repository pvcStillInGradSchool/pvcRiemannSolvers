// Copyright 2024 PEI Weicheng
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "mini/mesh/vtk.hpp"
#include "mini/mesh/part.hpp"
#include "mini/polynomial/projection.hpp"
#include "mini/polynomial/hexahedron.hpp"
#include "mini/polynomial/extrapolation.hpp"
#include "test/mesh/part.hpp"

class TestMeshVtk : public ::testing::Test {
 protected:
};
TEST_F(TestMeshVtk, EncodeBase64) {
  using mini::mesh::vtk::EncodeBase64;
  std::string origin, encoded;
  origin = "A";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "QQ==");
  origin = "AB";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "QUI=");
  origin = "ABC";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "QUJD");
  origin = "ABCD";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "QUJDRA==");
  origin = "ABCDE";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "QUJDREU=");
  origin = "ABCDEF";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "QUJDREVG");
  origin = "1";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "MQ==");
  origin = "12";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "MTI=");
  origin = "123";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "MTIz");
  origin = "1234";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "MTIzNA==");
  origin = "12345";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "MTIzNDU=");
  origin = "123456";
  encoded = EncodeBase64(origin.data(), origin.size());
  EXPECT_EQ(encoded, "MTIzNDU2");
}
TEST_F(TestMeshVtk, DecodeBase64) {
  using mini::mesh::vtk::DecodeBase64;
  std::string origin, decoded;
  origin = "A";
  decoded.resize(origin.size());
  DecodeBase64("QQ==", decoded.size(), decoded.data());
  EXPECT_EQ(origin, decoded);
  origin = "AB";
  decoded.resize(origin.size());
  DecodeBase64("QUI=", decoded.size(), decoded.data());
  EXPECT_EQ(origin, decoded);
  origin = "ABC";
  decoded.resize(origin.size());
  DecodeBase64("QUJD", decoded.size(), decoded.data());
  EXPECT_EQ(origin, decoded);
  origin = "ABCD";
  decoded.resize(origin.size());
  DecodeBase64("QUJDRA==", decoded.size(), decoded.data());
  EXPECT_EQ(origin, decoded);
  origin = "ABCDE";
  decoded.resize(origin.size());
  DecodeBase64("QUJDREU=", decoded.size(), decoded.data());
  EXPECT_EQ(origin, decoded);
  origin = "ABCDEF";
  decoded.resize(origin.size());
  DecodeBase64("QUJDREVG", decoded.size(), decoded.data());
  EXPECT_EQ(origin, decoded);
  origin = "1";
  decoded.resize(origin.size());
  DecodeBase64("MQ==", decoded.size(), decoded.data());
  EXPECT_EQ(origin, decoded);
  origin = "12";
  decoded.resize(origin.size());
  DecodeBase64("MTI=", decoded.size(), decoded.data());
  EXPECT_EQ(origin, decoded);
  origin = "123";
  decoded.resize(origin.size());
  DecodeBase64("MTIz", decoded.size(), decoded.data());
  EXPECT_EQ(origin, decoded);
  origin = "1234";
  decoded.resize(origin.size());
  DecodeBase64("MTIzNA==", decoded.size(), decoded.data());
  EXPECT_EQ(origin, decoded);
  origin = "12345";
  decoded.resize(origin.size());
  DecodeBase64("MTIzNDU=", decoded.size(), decoded.data());
  EXPECT_EQ(origin, decoded);
  origin = "123456";
  decoded.resize(origin.size());
  DecodeBase64("MTIzNDU2", decoded.size(), decoded.data());
  EXPECT_EQ(origin, decoded);
}

template <class Part>
void Write(std::string const &case_name, std::string const &solution_name) {
  auto part = Part(case_name, i_core, n_core);
  InstallIntegratorPrototypes(&part);
  part.SetFieldNames({"U1", "U2"});
  part.ReadSolutions(solution_name);
  part.ScatterSolutions();
  using VtkWriter = mini::mesh::vtk::Writer<Part>;
  using Cell = typename Part::Cell;
  auto plus = [](Cell const &cell) -> Scalar {
    Value value = cell.polynomial().average();
    return value[0] + value[1];
  };
  VtkWriter::AddCellData("U1+U2", plus);
  auto minus = [](Cell const &, Coord const &, Value const &value) -> Scalar {
    return value[0] - value[1];
  };
  VtkWriter::AddPointData("U1-U2", minus);
  VtkWriter::WriteSolutions(part, solution_name);
}

TEST_F(TestMeshVtk, Writer) {
  /* aproximated by Projection on OrthoNormal basis */
{
  auto solution_name = "Projection";
  std::printf("Run Write<%s>() on proc[%d/%d] at %f sec\n",
      solution_name, i_core, n_core, MPI_Wtime() - time_begin);
  using Projection = mini::polynomial::Projection<
      Scalar, kDimensions, kDegrees, kComponents>;
  using Part = mini::mesh::part::Part<cgsize_t, Projection>;
  Write<Part>("double_mach", solution_name);
}
  /* aproximated by Interpolation on Lagrange basis */
{
  auto solution_name = "Interpolation";
  std::printf("Run Write<%s>() on proc[%d/%d] at %f sec\n",
      solution_name, i_core, n_core, MPI_Wtime() - time_begin);
  using Interpolation = mini::polynomial::Hexahedron<Gx, Gx, Gx, kComponents, false>;
  using Extrapolation = mini::polynomial::Extrapolation<Interpolation>;
  using Part = mini::mesh::part::Part<cgsize_t, Extrapolation>;
  Write<Part>("double_mach", solution_name);
}
}

// mpirun -n 4 ./part must be run in ./
// mpirun -n 4 ./vtk
int main(int argc, char* argv[]) {
  return Main(argc, argv);
}
