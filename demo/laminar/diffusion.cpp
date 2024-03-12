//  Copyright 2024 PEI Weicheng
#include "mini/constant/index.hpp"
#include "mini/geometry/pi.hpp"
#include "shockless.hpp"

// length of the domain in each direction:
constexpr Scalar l_x{1.}, l_y{1.}, l_z{1.};

// number of half waves in each direction:
constexpr int n_x{2}, n_y{3}, n_z{4};

// wave number in each direction:
constexpr Scalar pi = mini::geometry::pi();
constexpr Scalar k_x = n_x * pi / l_x;
constexpr Scalar k_y = n_y * pi / l_y;
constexpr Scalar k_z = n_z * pi / l_z;

// viscosity and thermal conductivity:
constexpr Scalar half_life = 1.0;  // i.e. exp(-beta * half_life) == 0.5
constexpr Scalar beta = std::log(2.0) / half_life;
constexpr Scalar prandtl = 0.7;
constexpr Scalar nu = beta * prandtl / (Gas::Gamma() * k_x * k_x);

/* Set initial conditions. */
constexpr Scalar rho_given = 1.29;
constexpr Scalar u_given = 0.0;
constexpr Scalar v_given = 0.0;
constexpr Scalar w_given = 0.0;
constexpr Scalar p_given = 101325;
constexpr Scalar temperature_given = p_given / rho_given / Gas::R();

Value MyIC(const Global &xyz) {
  Scalar temperature = temperature_given;
  temperature +=
      std::cos(k_x * xyz[mini::constant::index::X]) *
      std::cos(k_y * xyz[mini::constant::index::Y]) *
      std::cos(k_z * xyz[mini::constant::index::Z]);
  Scalar pressure = rho_given * Gas::R() * temperature;
  Value value{ rho_given, 0, 0, 0, pressure / Gas::GammaMinusOne() };
  return value;
}

/* Set boundary conditions. */
Scalar temperature_gradient = 0.0;
auto fixed = [](const Global& xyz, double t){
  Value value;
  value[1] = 0;
  value[2] = 0;
  value[3] = 0;
  value[4] = temperature_gradient;
  return value;
};

void MyBC(const std::string &suffix, Spatial *spatial) {
  assert(suffix == "hexa");
  spatial->SetInviscidWall("4_S_26");   // Left
  spatial->SetInviscidWall("4_S_18");  // Right
  spatial->SetInviscidWall("4_S_22");  // Top
  spatial->SetInviscidWall("4_S_14");  // Bottom
  spatial->SetInviscidWall("4_S_5");   // Back
  spatial->SetInviscidWall("4_S_27");  // Front
}

int main(int argc, char* argv[]) {
  Riemann::SetBetaValues(2.0, 1.0 / 12);
  Riemann::SetProperty(nu, prandtl);
  std::cout << "b = " << nu * Gas::Gamma() / prandtl << std::endl;
  return Main(argc, argv, MyIC, MyBC);
}
