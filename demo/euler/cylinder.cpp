//  Copyright 2022 PEI Weicheng
#include "sourceless.hpp"

/* Set initial conditions. */
auto primitive = Primitive(1.4, 0.04, 0.03, 0.0, 1.0);
Value given_value = Gas::PrimitiveToConservative(primitive);

Value MyIC(const Global &xyz) {
  return given_value;
}

/* Set boundary conditions. */
auto given_state = [](const Global& xyz, double t){
  return given_value;
};

void MyBC(const std::string &suffix, Spatial *spatial) {
  spatial->SetSubsonicInlet("4_S_5", given_state);  // Left
  spatial->SetSubsonicInlet("4_S_4", given_state);  // Bottom
  spatial->SetSubsonicOutlet("4_S_6", given_state);  // Right
  spatial->SetSubsonicOutlet("4_S_7", given_state);  // Top
  spatial->SetInviscidWall("4_S_9");  // Front
  spatial->SetInviscidWall("4_S_3");  // Back
  spatial->SetInviscidWall("4_S_8");  // Cylinder
}

int main(int argc, char* argv[]) {
  return Main(argc, argv, MyIC, MyBC);
}
