//  Copyright 2023 PEI Weicheng
#include "sourceless.hpp"

/* Set initial conditions. */
auto primitive_left = Primitive(1.0, -4., 0.0, 0.0, 0.4);
auto primitive_right = Primitive(1.0, +4., 0.0, 0.0, 0.4);
auto value_left = Gas::PrimitiveToConservative(primitive_left);
auto value_right = Gas::PrimitiveToConservative(primitive_right);

Value MyIC(const Global &xyz) {
  auto x = xyz[0];
  return (x < 2.5) ? value_left : value_right;
}

/* Set boundary conditions. */
auto state_left = [](const Global& xyz, double t){ return value_left; };
auto state_right = [](const Global& xyz, double t) { return value_right; };

void MyBC(const std::string &suffix, Spatial *spatial) {
  if (suffix == "tetra") {
    spatial->SetSmartBoundary("3_S_42", state_left);  // Left
    spatial->SetSmartBoundary("3_S_20", state_right);  // Right
    spatial->SetInviscidWall("3_S_1");
    spatial->SetInviscidWall("3_S_2");
    spatial->SetInviscidWall("3_S_24");
    spatial->SetInviscidWall("3_S_38");
    spatial->SetInviscidWall("3_S_29");
    spatial->SetInviscidWall("3_S_51");
    spatial->SetInviscidWall("3_S_16");
    spatial->SetInviscidWall("3_S_46");
  } else {
    assert(suffix == "hexa");
    spatial->SetSmartBoundary("4_S_42", state_left);  // Left
    spatial->SetSmartBoundary("4_S_20", state_right);  // Right
    spatial->SetInviscidWall("4_S_1");
    spatial->SetInviscidWall("4_S_2");
    spatial->SetInviscidWall("4_S_24");
    spatial->SetInviscidWall("4_S_38");
    spatial->SetInviscidWall("4_S_29");
    spatial->SetInviscidWall("4_S_51");
    spatial->SetInviscidWall("4_S_16");
    spatial->SetInviscidWall("4_S_46");
  }
}

int main(int argc, char* argv[]) {
  return Main(argc, argv, MyIC, MyBC);
}
