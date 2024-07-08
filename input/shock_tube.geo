/*
 * Generate unstructured mesh for "shock tube" problems with exact initial conditions.
 */
LC = 0.1;  // average length of cell edges, a.k.a. characteristic length
LX = 2.5;  // length along x-axis
LY = 2 * LC;  // length along y-axis
LZ = 3 * LC;  // length along z-axis

Point(1) = { LX, 0., 0., LC };
Point(2) = { LX+LX, 0., 0., LC };
Point(3) = { LX+LX, LY, 0., LC };
Point(4) = { LX, LY, 0., LC };
Point(5) = { 0., LY, 0., LC };
Point(6) = { 0., 0., 0., LC };

Line(1) = { 1, 2 };
Line(2) = { 2, 3 };
Line(3) = { 3, 4 };
Line(4) = { 4, 1 };
Line(5) = { 4, 5 };
Line(6) = { 5, 6 };
Line(7) = { 6, 1 };

Curve Loop(1) = { 1, 2, 3, 4 };
Curve Loop(2) = { 5, 6, 7, -4 };
Plane Surface(1) = { 1 };
Plane Surface(2) = { 2 };
Recombine Surface{ 1, 2 };
out[] = Extrude{ 0, 0, LZ }{
  Surface{ 1, 2 }; Layers{ Ceil(LZ/LC/2)*2 - 1 };  Recombine;
};

Physical Surface("Left") = { 42 };
Physical Surface("Right") = { 20 };
Physical Surface("Wall") = { 1, 2, 24, 38, 29, 51, 16, 46 };
Physical Volume("Fluid") = { 1, 2 };

Mesh 3;
