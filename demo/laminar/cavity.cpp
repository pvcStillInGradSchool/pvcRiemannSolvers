//  Copyright 2024 PEI Weicheng
#include "mini/constant/index.hpp"
#include "mini/geometry/pi.hpp"
#include "shockless.hpp"

/* Set initial conditions. */
Scalar rho_given = 1.0;
Scalar angle = 0.0;
auto cos_sin = mini::geometry::CosSin(angle);
Scalar u_given = rho_given * cos_sin.first;
Scalar v_given = rho_given * 0.0;
Scalar w_given = rho_given * cos_sin.second;
Scalar p_given = 101325 / 1.29;

Value MyIC(const Global &xyz) {
  auto primitive = Primitive(rho_given, u_given, v_given, w_given, p_given);
  if (xyz[mini::constant::index::Y] < 0.9999) {
    primitive.momentum().setZero();
  }
  Value value = Gas::PrimitiveToConservative(primitive);
  return value;
}

/* Set boundary conditions. */
Scalar temperature_gradient = 0.0;
auto moving = [](const Global& xyz, double t){
  Value value;
  value[1] = u_given;
  value[2] = v_given;
  value[3] = w_given;
  value[4] = temperature_gradient;
  return value;
};
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
  spatial->SetNoSlipWall("4_S_26", fixed);   // Left
  spatial->SetNoSlipWall("4_S_18", fixed);  // Right
  spatial->SetNoSlipWall("4_S_22", moving);  // Top
  spatial->SetNoSlipWall("4_S_14", fixed);  // Bottom
  spatial->SetInviscidWall("4_S_5");   // Back
  spatial->SetInviscidWall("4_S_27");  // Front
}

int main(int argc, char* argv[]) {
  Riemann::SetBetaValues(2.0, 1.0 / 12);
  Riemann::SetProperty(/* nu = */0.01, /* prandtl = */0.708);
  return Main(argc, argv, MyIC, MyBC);
}
