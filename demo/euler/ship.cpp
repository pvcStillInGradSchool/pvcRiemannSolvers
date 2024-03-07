//  Copyright 2022 PEI Weicheng
#include "sourceless.hpp"

/* Set initial conditions. */
auto primitive = Primitive(1.29, 5.0, 0.0, 0.0, 101325.0);
Value given_value = Gas::PrimitiveToConservative(primitive);

Value MyIC(const Global &xyz) {
  return given_value;
}

/* Set boundary conditions. */
auto given_state = [](const Global& xyz, double t){
  return given_value;
};

void MyBC(const std::string &suffix, Spatial *spatial) {
  // bounding box
  spatial->SetSmartBoundary("3_S_37"/* Front */, given_state);
  spatial->SetSmartBoundary("3_S_36"/* Left */, given_state);
  spatial->SetSmartBoundary("3_S_41"/* Right */, given_state);
  spatial->SetSmartBoundary("3_S_38"/* Top */, given_state);
  spatial->SetSmartBoundary("3_S_39"/* Back */, given_state);
  // bottom
  spatial->SetInviscidWall("3_S_27"/* fine */);
  spatial->SetInviscidWall("3_S_40"/* coarse */);
  // tower
  spatial->SetInviscidWall("3_S_13");
  spatial->SetInviscidWall("3_S_14");
  spatial->SetInviscidWall("3_S_15");
  spatial->SetInviscidWall("3_S_16");
  spatial->SetInviscidWall("3_S_18");
  // nose
  spatial->SetInviscidWall("3_S_21");
  spatial->SetInviscidWall("3_S_22");
  spatial->SetInviscidWall("3_S_23");
  // body
  spatial->SetInviscidWall("3_S_28");
  spatial->SetInviscidWall("3_S_29");
  spatial->SetInviscidWall("3_S_30");
  spatial->SetInviscidWall("3_S_35");
  spatial->SetInviscidWall("3_S_42");
  spatial->SetInviscidWall("3_S_43");
  spatial->SetInviscidWall("3_S_44");
  spatial->SetInviscidWall("3_S_45");
  // deck
  spatial->SetInviscidWall("3_S_2");
  spatial->SetInviscidWall("3_S_6");
  spatial->SetInviscidWall("3_S_33");
  spatial->SetInviscidWall("3_S_34");
}

int main(int argc, char* argv[]) {
  return Main(argc, argv, MyIC, MyBC);
}
