//  Copyright 2022 PEI Weicheng
#include "sourceless.hpp"
#include "mini/geometry/pi.hpp"

/* Set initial conditions. */
auto [c, s] = mini::geometry::CosSin(20.0);
Value upstream_value = Gas::PrimitiveToConservative(
    Primitive(1.4, 0.3 * c, 0.0, 0.3 * s, 1.0));
Value MyIC(const Global &xyz) {
  return upstream_value;
}

/* Set boundary conditions. */
Value exhaust_value = Gas::PrimitiveToConservative(
    Primitive(1.4, 2.4, 0.0, 0.0, 1.44));
auto exhaust = [](const Global& xyz, double t){
  return exhaust_value;
};
auto upstream = [](const Global& xyz, double t){
  return upstream_value;
};
void MyBC(const std::string &suffix, Spatial *spatial) {
  spatial->SetSubsonicInlet("upstream", upstream);
  spatial->SetSupersonicInlet("exhaust", exhaust);
  spatial->SetSubsonicOutlet("downstream", upstream);
  spatial->SetSubsonicOutlet("intake", upstream);
  spatial->SetInviscidWall("intake ramp");
  spatial->SetSubsonicInlet("lower", upstream);
  spatial->SetSubsonicOutlet("upper", upstream);
  spatial->SetInviscidWall("strake");
  spatial->SetInviscidWall("vertical tail");
  spatial->SetInviscidWall("horizontal tail");
  spatial->SetInviscidWall("side");
  spatial->SetInviscidWall("wing");
  spatial->SetInviscidWall("fuselage");
  spatial->SetInviscidWall("symmetry");
}

int main(int argc, char* argv[]) {
  return Main(argc, argv, MyIC, MyBC);
}
