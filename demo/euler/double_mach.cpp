//  Copyright 2022 PEI Weicheng
#include "sourceless.hpp"

/* Set initial conditions. */
constexpr double kGammaPlus = 2.4, kGamma = 1.4, kGammaMinus = 0.4;
double rho_before = 1.4, p_before = 1.0;
double m_before = 10.0, a_before = 1.0, u_nu = m_before * a_before;
auto rho_after = rho_before * (m_before * m_before * kGammaPlus / 2.0)
    / (1.0 + m_before * m_before * kGammaMinus / 2.0);  // 8.0
auto p_after = p_before * (m_before * m_before * kGamma - kGammaMinus / 2.0)
    / (kGammaPlus / 2.0);  // 116.5
auto u_n_after = u_nu * (rho_after - rho_before) / rho_after;  // 8.25
auto tan_60 = std::sqrt(3.0), cos_30 = tan_60 * 0.5, sin_30 = 0.5;
auto u_after = u_n_after * cos_30, v_after = u_n_after * (-sin_30);
auto primitive_after = Primitive(rho_after, u_after, v_after, 0.0, p_after);
auto primitive_before = Primitive(rho_before, 0.0, 0.0, 0.0, p_before);
Value value_after = Gas::PrimitiveToConservative(primitive_after);
Value value_before = Gas::PrimitiveToConservative(primitive_before);
double x_gap = 1.0 / 6.0;

Value MyIC(const Global &xyz) {
  auto x = xyz[0], y = xyz[1];
  return ((x - x_gap) * tan_60 < y) ? value_after : value_before;
}

/* Set boundary conditions. */
auto u_x = u_nu / cos_30;
auto moving_shock = [](const Global& xyz, double t){
  auto x = xyz[0], y = xyz[1];
  return ((x - (x_gap + u_x * t)) * tan_60 < y) ? value_after : value_before;
};

void MyBC(const std::string &suffix, Spatial *spatial) {
  if (suffix == "tetra") {
    spatial->SetSupersonicInlet("3_S_27", moving_shock);  // Top
    spatial->SetSupersonicInlet("3_S_31", moving_shock);  // Left
    spatial->SetInviscidWall("3_S_1");   // Back
    spatial->SetInviscidWall("3_S_32");  // Front
    spatial->SetInviscidWall("3_S_19");  // Bottom
    spatial->SetSupersonicOutlet("3_S_23");  // Right
    spatial->SetSupersonicOutlet("3_S_15");  // Gap
  } else {
    assert(suffix == "hexa");
    spatial->SetSupersonicInlet("4_S_27", moving_shock);  // Top
    spatial->SetSupersonicInlet("4_S_31", moving_shock);  // Left
    spatial->SetInviscidWall("4_S_1");   // Back
    spatial->SetInviscidWall("4_S_32");  // Front
    spatial->SetInviscidWall("4_S_19");  // Bottom
    spatial->SetSupersonicOutlet("4_S_23");  // Right
    spatial->SetSupersonicOutlet("4_S_15");  // Gap
  }
}

int main(int argc, char* argv[]) {
  return Main(argc, argv, MyIC, MyBC);
}
