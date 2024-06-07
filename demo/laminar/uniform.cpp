//  Copyright 2024 PEI Weicheng
#include "mini/constant/index.hpp"
#include "mini/geometry/pi.hpp"
#include "shockless.hpp"

constexpr Scalar mach_infty = 0.2;
constexpr Scalar total_pressure_infty = 104165;
constexpr Scalar total_temperature_infty = 290.304;
constexpr Scalar pressure_infty = Gas::TotalPressureToPressure(
    mach_infty, total_pressure_infty);  // 101300 Pa
constexpr Scalar temperature_infty = Gas::TotalTemperatureToTemperature(
    mach_infty, total_temperature_infty);
constexpr Scalar density_infty = pressure_infty / temperature_infty / Gas::R();
constexpr Scalar uvw_infty = mach_infty *
    Gas::GetSpeedOfSound(density_infty, pressure_infty);
constexpr Scalar v_cos = 0.3, w_cos = 0.4;
constexpr Scalar u_cos = std::sqrt(1 - v_cos * v_cos - w_cos * w_cos);
static_assert(std::abs(u_cos * u_cos + v_cos * v_cos + w_cos * w_cos - 1) < 1e-8);
constexpr Scalar u_infty = uvw_infty * u_cos;
constexpr Scalar v_infty = uvw_infty * v_cos;
constexpr Scalar w_infty = uvw_infty * w_cos;

/* Set initial conditions. */

Value MyIC(const Global &xyz) {
  Primitive primitive{density_infty, u_infty, v_infty, w_infty, pressure_infty};
  return Gas::PrimitiveToConservative(primitive);
}

/* Set boundary conditions. */
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

void MyBC(const std::string &suffix, Spatial *spatial) {
  assert(suffix == "hexa");
  spatial->SetSubsonicInlet("4_S_26", inlet);   // Left
  spatial->SetSubsonicInlet("4_S_14", inlet);  // Bottom
  spatial->SetSubsonicInlet("4_S_5", inlet);   // Back
  spatial->SetSubsonicOutlet("4_S_18", outlet);  // Right
  spatial->SetSubsonicOutlet("4_S_22", outlet);  // Top
  spatial->SetSubsonicOutlet("4_S_27", outlet);  // Front
}

int main(int argc, char* argv[]) {
  Riemann::Diffusion::SetBetaValues(2.0, 1.0 / 12);
  Riemann::Diffusion::SetProperty(/* nu = */0.00, /* prandtl = */0.708);
  return Main(argc, argv, MyIC, MyBC);
}
