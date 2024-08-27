//  Copyright 2024 PEI Weicheng
#include "mini/constant/index.hpp"
#include "mini/geometry/pi.hpp"
#include "shocked.hpp"

/* Set initial conditions. */
auto primitive_left = Primitive(120., 0.0, 0.0, 0.0, 120. / Gas::Gamma());
auto primitive_right = Primitive(1.2, 0.0, 0.0, 0.0, 1.2 / Gas::Gamma());
auto value_left = Gas::PrimitiveToConservative(primitive_left);
auto value_right = Gas::PrimitiveToConservative(primitive_right);

Value MyIC(const Global &xyz) {
  auto x = xyz[mini::constant::index::X];
  return (x < 0.5) ? value_left : value_right;
};

/* Set boundary conditions. */
auto wall = [](const Global& xyz, double t){
  Value value;
  value[1] = 0;
  value[2] = 0;
  value[3] = 0;
  value[4] = 0;  // interpreted as temperature gradient
  return value;
};

void MyBC(const std::string &suffix, Spatial *spatial) {
  auto prefix = std::string(suffix == "tetra" ? "3_" : "4_");
  spatial->SetInviscidWall(prefix + "S_1");   // Back
  spatial->SetInviscidWall(prefix + "S_2");   // Back
  spatial->SetInviscidWall(prefix + "S_3");   // Back
  spatial->SetInviscidWall(prefix + "S_4");   // Back
  spatial->SetInviscidWall(prefix + "S_34");  // Front
  spatial->SetInviscidWall(prefix + "S_56");  // Front
  spatial->SetInviscidWall(prefix + "S_78");  // Front
  spatial->SetInviscidWall(prefix + "S_100");  // Front
  spatial->SetNoSlipWall(prefix + "S_33", wall);  // Left
  spatial->SetNoSlipWall(prefix + "S_95", wall);  // Left
  spatial->SetNoSlipWall(prefix + "S_47", wall);  // Right
  spatial->SetNoSlipWall(prefix + "S_69", wall);  // Right
  spatial->SetNoSlipWall(prefix + "S_73", wall);  // Top
  spatial->SetNoSlipWall(prefix + "S_91", wall);  // Top
  spatial->SetNoSlipWall(prefix + "S_21", wall);  // Bottom
  spatial->SetNoSlipWall(prefix + "S_43", wall);  // Bottom
}

int main(int argc, char* argv[]) {
  RiemannWithViscosity::Diffusion::SetBetaValues(2.0, 1.0 / 12);
  Diffusion::SetProperty(/* nu = */5e-3, /* prandtl = */0.73);
  return Main(argc, argv, MyIC, MyBC);
}
