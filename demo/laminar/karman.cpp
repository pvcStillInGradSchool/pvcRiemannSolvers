//  Copyright 2024 PEI Weicheng
#include "mini/constant/index.hpp"
#include "mini/geometry/pi.hpp"
#include "shockless.hpp"

Scalar diameter = 2.0;
Scalar reynolds = 120;

/* Set initial conditions. */
Scalar mach_infty = 0.2;
Scalar total_pressure_infty = 104165;
Scalar total_temperature_infty = 290.304;
auto pressure_infty = Gas::TotalPressureToPressure(
    mach_infty, total_pressure_infty);  // 101300 Pa
auto temperature_infty = Gas::TotalTemperatureToTemperature(
    mach_infty, total_temperature_infty);
auto density_infty = pressure_infty / temperature_infty / Gas::R();
auto u_infty = mach_infty *
    Gas::GetSpeedOfSound(density_infty, pressure_infty);
auto kinetic_viscosity = u_infty * diameter / reynolds;

Value MyIC(const Global &xyz) {
  auto primitive = Primitive(density_infty,
      u_infty, 0.0, 0.0, pressure_infty);
  Value value = Gas::PrimitiveToConservative(primitive);
  return value;
};

/* Set boundary conditions. */
auto outlet = [](const Global& xyz, double t){
  Value value;
  value[4] = pressure_infty;
  return value;
};
auto far_field = [](const Global& xyz, double t){
  auto primitive = Primitive(density_infty,
      u_infty, 0.0, 0.0, pressure_infty);
  Value value = Gas::PrimitiveToConservative(primitive);
  return value;
};
auto solid_wall = [](const Global& xyz, double t){
  Value value;
  value[1] = 0;
  value[2] = 0;
  value[3] = 0;
  value[4] = 0;  // interpreted as temperature gradient
  return value;
};

void MyBC(const std::string &suffix, Spatial *spatial) {
  assert(suffix == "hexa");
  spatial->SetSmartBoundary("4_S_4", far_field);  // Left
  spatial->SetSmartBoundary("4_S_5", far_field);  // Right
  spatial->SetSmartBoundary("4_S_6", far_field);  // Top
  spatial->SetSmartBoundary("4_S_3", far_field);  // Bottom
  spatial->SetNoSlipWall("4_S_53", solid_wall);  // Wall
  // Back:
  spatial->SetInviscidWall("4_S_1");
  spatial->SetInviscidWall("4_S_2");
  spatial->SetInviscidWall("4_S_52");
  // Front:
  spatial->SetInviscidWall("4_S_11");
  spatial->SetInviscidWall("4_S_13");
  spatial->SetInviscidWall("4_S_44");
}

int main(int argc, char* argv[]) {
  Riemann::Diffusion::SetBetaValues(2.0, 1.0 / 12);
  Riemann::Diffusion::SetProperty(/* nu = */kinetic_viscosity, /* prandtl = */0.708);
  return Main(argc, argv, MyIC, MyBC);
}
