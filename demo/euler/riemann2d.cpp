//  Copyright 2024 PEI Weicheng
#include "sourceless.hpp"

/* Set initial conditions. */
auto primitive_a = Primitive(1.5, 0.0, 0.0, 0.0, 1.5);
auto primitive_b = Primitive(0.5323, 1.206, 0.0, 0.0, 0.3);
auto primitive_c = Primitive(0.1380, 1.206, 1.206, 0.0, 0.029);
auto primitive_d = Primitive(0.5323, 0.0, 1.206, 0.0, 0.3);
auto value_a = Gas::PrimitiveToConservative(primitive_a);
auto value_b = Gas::PrimitiveToConservative(primitive_b);
auto value_c = Gas::PrimitiveToConservative(primitive_c);
auto value_d = Gas::PrimitiveToConservative(primitive_d);

// Get the speed of shocks by R-H:
Scalar const speed_cd = (value_d.momentumX() - value_c.momentumX())
    / (primitive_d.rho() - primitive_c.rho());
Scalar const speed_bc = (value_c.momentumY() - value_b.momentumY())
    / (primitive_c.rho() - primitive_b.rho());

Value MyIC(const Global &xyz) {
  auto x = xyz[0], y = xyz[1];
  return x < 0.8
      ? (y < 0.8 ? value_c : value_b)
      : (y < 0.8 ? value_d : value_a);
}

/* Set boundary conditions. */
auto state_left = [](const Global& xyz, double t) {
  return (xyz[1] - speed_bc * t) < 0.8 ? value_c : value_b;
};
auto state_bottom = [](const Global& xyz, double t) {
  return (xyz[0] - speed_cd * t) < 0.8 ? value_c : value_d;
};

void MyBC(const std::string &suffix, Spatial *spatial) {
  auto prefix = std::string(suffix == "tetra" ? "3_" : "4_");
  spatial->SetInviscidWall(prefix + "S_73");  // Top
  spatial->SetInviscidWall(prefix + "S_91");  // Top
  spatial->SetInviscidWall(prefix + "S_1");   // Back
  spatial->SetInviscidWall(prefix + "S_2");   // Back
  spatial->SetInviscidWall(prefix + "S_3");   // Back
  spatial->SetInviscidWall(prefix + "S_4");   // Back
  spatial->SetInviscidWall(prefix + "S_34");  // Front
  spatial->SetInviscidWall(prefix + "S_56");  // Front
  spatial->SetInviscidWall(prefix + "S_78");  // Front
  spatial->SetInviscidWall(prefix + "S_100");  // Front
  spatial->SetInviscidWall(prefix + "S_47");  // Right
  spatial->SetInviscidWall(prefix + "S_69");  // Right
  spatial->SetSmartBoundary(prefix + "S_33", state_left);  // Left
  spatial->SetSmartBoundary(prefix + "S_95", state_left);  // Left
  spatial->SetSmartBoundary(prefix + "S_21", state_bottom);  // Bottom
  spatial->SetSmartBoundary(prefix + "S_43", state_bottom);  // Bottom
}

int main(int argc, char* argv[]) {
  return Main(argc, argv, MyIC, MyBC);
}
