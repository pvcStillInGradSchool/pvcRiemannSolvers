/*
 * Generate a structured hexahedral mesh in a stretched box.
 */
LC = 0.1;  // average length of cell edges, a.k.a. characteristic length
LX = 4.0;  // length along x-axis
LY = 1.0;  // length along y-axis
LZ = 0.3;  // length along z-axis

Point(1) = { 0., 0., 0., LC };
Point(2) = { LX, 0., 0., LC };
Point(3) = { LX, LY, 0., LC };
Point(4) = { 0., LY, 0., LC };

Line(1) = { 1, 2 };
Line(2) = { 2, 3 };
Line(3) = { 3, 4 };
Line(4) = { 4, 1 };

Curve Loop(1) = { 1, 2, 3, 4 };
Plane Surface(1) = { 1 };
Transfinite Curve{ 1, 3 } = LX/LC + 1.01;
Transfinite Curve{ 2, 4 } = LY/LC + 1.01;
Transfinite Surface{ 1 };
Recombine Surface{ 1 };
out[] = Extrude{ 0, 0, LZ }{
  Surface{ 1 }; Layers{ LZ/LC + 0.01 }; Recombine;
};

Physical Surface("Back") = { 1 };
Physical Surface("Front") = { 26 };
Physical Surface("Left") = { 25 };
Physical Surface("Right") = { 17 };
Physical Surface("Top") = { 21 };
Physical Surface("Bottom") = { 13 };
Physical Volume("Volume") = { 1 };

Mesh 3;
