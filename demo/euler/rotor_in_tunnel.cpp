//  Copyright 2022 PEI Weicheng
#include "rotorcraft.hpp"

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
  spatial->SetSubsonicInlet("3_S_13"/* Left */, given_state);
  spatial->SetInviscidWall("3_S_14"/* Bottom */);
  spatial->SetInviscidWall("3_S_15"/* Front */);
  spatial->SetInviscidWall("3_S_16"/* Top */);
  spatial->SetInviscidWall("3_S_17"/* Back */);
  spatial->SetSubsonicOutlet("3_S_18"/* Right */, given_state);
}

int main(int argc, char* argv[]) {
  auto source = Source();
  auto kOmega = 35.0;
  // build a blade
  std::vector<double> y_values{0.0, 0.9},
      chords{0.1, 0.1},
      twists{+10.0, +10.0};
  auto sc1095 = mini::aircraft::airfoil::SC1095<Scalar>();
  auto linear = mini::aircraft::airfoil::Linear<Scalar>(0.1, 0);
  std::vector<mini::aircraft::airfoil::Abstract<Scalar> const *>
      airfoils{ &sc1095, &linear };
  auto blade = Blade();
  for (int i = 0, n = y_values.size(); i < n; ++i) {
    blade.InstallSection(y_values[i], chords[i], twists[i], *airfoils[0]);
  }
  double root{0.1};
{
  // Set parameters for the 1st rotor:
  auto rotor = Rotor();
  rotor.SetRevolutionsPerSecond(+kOmega);  // right-hand rotation
  rotor.SetOrigin(1.0, 0.0, 1.0);
  auto frame = Frame();
  frame.RotateY(-5.0/* deg */);
  rotor.SetFrame(frame);
  rotor.InstallBlade(root, blade);
  rotor.InstallBlade(root, blade);
  rotor.InstallBlade(root, blade);
  rotor.SetInitialAzimuth(0.0);
  source.InstallRotor(rotor);
}
{
  // Set parameters for the 2nd rotor:
  auto rotor = Rotor();
  rotor.SetRevolutionsPerSecond(-kOmega);  // left-hand rotation
  rotor.SetOrigin(1.0, 0.0, -1.0);
  auto frame = Frame();
  frame.RotateY(-5.0/* deg */);
  rotor.SetFrame(frame);
  rotor.InstallBlade(root, blade);
  rotor.InstallBlade(root, blade);
  rotor.InstallBlade(root, blade);
  rotor.SetInitialAzimuth(0.0);
  source.InstallRotor(rotor);
}

  return Main(argc, argv, MyIC, MyBC, source);
}
