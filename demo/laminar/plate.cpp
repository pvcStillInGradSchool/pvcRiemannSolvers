//  Copyright 2024 PEI Weicheng
#include "mini/constant/index.hpp"
#include "mini/geometry/pi.hpp"
#include "shockless.hpp"

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

Value MyIC(const Global &xyz) {
  auto x = xyz[mini::constant::index::X];
  auto y = xyz[mini::constant::index::Y];
  auto u_bottom = u_infty * (1 - x / 0.2);
  auto du_dy = (u_infty - u_bottom) / 0.02;
  auto u = du_dy * y + u_bottom;
  auto primitive = Primitive(density_infty,
      u, 0.0, 0.0, pressure_infty);
  Value value = Gas::PrimitiveToConservative(primitive);
  return value;
};

/* Set boundary conditions. */
Scalar u_cos = 1.0, v_cos = 0.0, w_cos = 0.0;
auto inlet = [](const Global& xyz, double t){
  Value value;
  value[0] = total_pressure_infty;
  value[1] = u_cos;
  value[2] = v_cos;
  value[3] = w_cos;
  value[4] = total_temperature_infty;
  return value;
};
auto outlet = [](const Global& xyz, double t){
  Value value;
  value[4] = pressure_infty;
  return value;
};
auto bottom = [](const Global& xyz, double t){
  Value value;
  value[1] = 0;
  value[2] = 0;
  value[3] = 0;
  value[4] = 0;  // interpreted as temperature gradient
  return value;
};

void MyBC(const std::string &suffix, Spatial *spatial) {
  assert(suffix == "hexa");
  spatial->SetSubsonicInlet("4_S_26", inlet);   // Left
  spatial->SetSubsonicOutlet("4_S_18", outlet);  // Right
  spatial->SetSubsonicOutlet("4_S_22", outlet);  // Top
  spatial->SetNoSlipWall("4_S_14", bottom);  // Bottom
  spatial->SetInviscidWall("4_S_5");   // Back
  spatial->SetInviscidWall("4_S_27");  // Front
}

int main(int argc, char* argv[]) {
  Riemann::SetBetaValues(2.0, 1.0 / 12);
  Riemann::SetProperty(/* nu = */1.57e-5, /* prandtl = */0.708);
  return Main(argc, argv, MyIC, MyBC);
}