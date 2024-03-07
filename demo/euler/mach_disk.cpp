//  Copyright 2022 PEI Weicheng
#include "sourceless.hpp"
#include "mini/geometry/pi.hpp"

/* Set initial conditions. */
Value ambient_value = Gas::PrimitiveToConservative(
    Primitive(1.4, 0.3, 0.0, 0.0, 1.0));
Value MyIC(const Global &xyz) {
  return ambient_value;
}

/* Set boundary conditions. */
Value exhaust_value = Gas::PrimitiveToConservative(
    Primitive(1.4, 2.4, 0.0, 0.0, 1.44));
auto exhaust = [](const Global& xyz, double t){
  return exhaust_value;
};
auto ambient = [](const Global& xyz, double t){
  return ambient_value;
};
void MyBC(const std::string &suffix, Spatial *spatial) {
  spatial->SetSupersonicInlet("3_S_8", exhaust);
  spatial->SetSubsonicInlet("3_S_1", ambient);
  spatial->SetSupersonicOutlet("3_S_7");
  spatial->SetInviscidWall("3_S_2");
  spatial->SetInviscidWall("3_S_3");
  spatial->SetInviscidWall("3_S_4");
  spatial->SetInviscidWall("3_S_5");
  spatial->SetInviscidWall("3_S_6");
}

int main(int argc, char* argv[]) {
  return Main(argc, argv, MyIC, MyBC);
}
