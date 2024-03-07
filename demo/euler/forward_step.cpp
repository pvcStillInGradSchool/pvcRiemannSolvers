//  Copyright 2022 PEI Weicheng
#include "sourceless.hpp"

/* Set initial conditions. */
auto primitive = Primitive(1.4, 3.0, 0.0, 0.0, 1.0);
Value given_value = Gas::PrimitiveToConservative(primitive);

Value MyIC(const Global &xyz) {
  return given_value;
}

/* Set boundary conditions. */
auto given_state = [](const Global& xyz, double t){ return given_value; };

void MyBC(const std::string &suffix, Spatial *spatial) {
  if (suffix == "tetra") {
    spatial->SetSupersonicInlet("3_S_53", given_state);  // Left-Upper
    spatial->SetSupersonicInlet("3_S_31", given_state);  // Left-Lower
    spatial->SetInviscidWall("3_S_49"); spatial->SetInviscidWall("3_S_71");  // Top
    spatial->SetInviscidWall("3_S_1"); spatial->SetInviscidWall("3_S_2");
    spatial->SetInviscidWall("3_S_3");  // Back
    spatial->SetInviscidWall("3_S_54"); spatial->SetInviscidWall("3_S_76");
    spatial->SetInviscidWall("3_S_32");  // Front
    spatial->SetInviscidWall("3_S_19"); spatial->SetInviscidWall("3_S_23");
    spatial->SetInviscidWall("3_S_63");  // Step
    spatial->SetSupersonicOutlet("3_S_67");  // Right
  } else {
    assert(suffix == "hexa");
    spatial->SetSupersonicInlet("4_S_53", given_state);  // Left-Upper
    spatial->SetSupersonicInlet("4_S_31", given_state);  // Left-Lower
    spatial->SetInviscidWall("4_S_49"); spatial->SetInviscidWall("4_S_71");  // Top
    spatial->SetInviscidWall("4_S_1"); spatial->SetInviscidWall("4_S_2");
    spatial->SetInviscidWall("4_S_3");  // Back
    spatial->SetInviscidWall("4_S_54"); spatial->SetInviscidWall("4_S_76");
    spatial->SetInviscidWall("4_S_32");  // Front
    spatial->SetInviscidWall("4_S_19"); spatial->SetInviscidWall("4_S_23");
    spatial->SetInviscidWall("4_S_63");  // Step
    spatial->SetSupersonicOutlet("4_S_67");  // Right
  }
}

int main(int argc, char* argv[]) {
  return Main(argc, argv, MyIC, MyBC);
}
