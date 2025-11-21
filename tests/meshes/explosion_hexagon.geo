// Gmsh project created on Thu Nov 20 15:11:21 2025
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 2.0, 0, 1.0};
//+
Point(2) = {1, 2.0, 0, 1.0};
//+
Point(3) = {-1, 2.0, 0, 1.0};
//+
Point(4) = {-2, 0, 0, 1.0};
//+
Point(5) = {-1, -2, 0, 1.0};
//+
Point(6) = {0, -2, 0, 1.0};
//+
Point(7) = {1, -2, 0, 1.0};
//+
Point(8) = {2, 0, 0, 1.0};
//+
Line(1) = {4, 3};
//+
Line(2) = {3, 1};
//+
Line(3) = {1, 2};
//+
Line(4) = {2, 8};
//+
Line(5) = {8, 7};
//+
Line(6) = {7, 6};
//+
Line(7) = {6, 5};
//+
Line(8) = {5, 4};
//+
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};
//+
Plane Surface(1) = {1};
//+
Physical Curve(1007) = {1, 2, 3, 4, 5, 6, 7, 8};

Physical Surface(1) = {1};

// Settings
// This value gives the global element size factor (lower -> finer mesh)
Mesh.CharacteristicLengthFactor = 1.0 * 2^(-3);
// Insist on quads instead of default triangles
Mesh.RecombineAll = 1;
Mesh.RecombinationAlgorithm = 2;
// Violet instead of green base color for better visibility
Mesh.ColorCarousel = 0;
