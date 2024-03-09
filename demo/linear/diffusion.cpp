//  Copyright 2024 PEI Weicheng
#include "mini/constant/index.hpp"
#include "mini/geometry/pi.hpp"
#include "system.hpp"

/* Set initial conditions. */

// length of the domain in each direction:
constexpr Scalar l_x{1.}, l_y{1.}, l_z{1.};

// number of half waves in each direction:
constexpr int n_x{2}, n_y{3}, n_z{4};

// wave number in each direction:
constexpr Scalar pi = mini::geometry::pi();
constexpr Scalar k_x = n_x * pi / l_x;
constexpr Scalar k_y = n_y * pi / l_y;
constexpr Scalar k_z = n_z * pi / l_z;

Value MyIC(const Global &xyz) {
  Scalar u = std::sin(k_x * xyz[mini::constant::index::X]);
  Scalar v = std::sin(k_y * xyz[mini::constant::index::Y]);
  Scalar w = std::sin(k_z * xyz[mini::constant::index::Z]);
  Value value{ u, v, w };
  return value;
}

/* Set boundary conditions. */
auto fixed = [](const Global& xyz, double t){
  Value value;
  value.setZero();
  return value;
};

void MyBC(const std::string &suffix, Spatial *spatial) {
  assert(suffix == "hexa");
  spatial->SetNoSlipWall("4_S_26", fixed);   // Left
  spatial->SetNoSlipWall("4_S_18", fixed);  // Right
  spatial->SetNoSlipWall("4_S_22", fixed);  // Top
  spatial->SetNoSlipWall("4_S_14", fixed);  // Bottom
  spatial->SetNoSlipWall("4_S_5", fixed);   // Back
  spatial->SetNoSlipWall("4_S_27", fixed);  // Front
}

int main(int argc, char* argv[]) {
  Riemann::SetBetaValues(2.0, 1.0 / 12);
  Riemann::SetDiffusionCoefficient(/* nu = */0.05);
  Jacobian a; a.setZero();
  Riemann::SetConvectionCoefficient(a, a, a);
  return Main(argc, argv, MyIC, MyBC);
}
