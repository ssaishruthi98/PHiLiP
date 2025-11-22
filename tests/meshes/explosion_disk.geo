// Gmsh project created on Thu Nov 20 15:11:21 2025
SetFactory("OpenCASCADE");
//+
h = 1.0;
R=2.0;
r=0.3;
c1 = 1.4;
c2 = 0.98994949366;
r2 = 0.3;

// Center Points of circle
Point(1) = {0, 0, 0, h}; 
Point(2) = {c1, 0, 0, h};
Point(3) = {-c1, 0, 0, h};
Point(4) = {0,c1, 0, h};
Point(5) = {0,-c1, 0, h};
Point(6) = {c2, c2, 0, h};
Point(7) = {c2, -c2, 0, h};
Point(8) = {-c2, c2, 0, h};
Point(9) = {-c2, -c2, 0, h};
// Points on circles
Point(10) = {2, 0, 0, h};            // p1
Point(11) = {-2, 0, 0, h};           // p1
Point(12) = {c1+r, 0, 0, h};         // p2
Point(13) = {c1-r, 0, 0, h};         // p2
Point(14) = {-c1+r, 0, 0, h};        // p3
Point(15) = {-c1-r, 0, 0, h};        // p3
Point(16) = {0,c1+r, 0, h};          // p4
Point(17) = {0,c1-r, 0, h};          // p4
Point(18) = {0,-c1+r, 0, h};         // p5
Point(19) = {0,-c1-r, 0, h};         // p5
Point(20) = {c2, c2+r2, 0, h};       // p6
Point(21) = {c2, c2-r2, 0, h};       // p6
Point(22) = {c2, -c2+r2, 0, h};      // p7
Point(23) = {c2, -c2-r2, 0, h};      // p7
Point(24) = {-c2, c2+r2, 0, h};      // p8
Point(25) = {-c2, c2-r2, 0, h};      // p8
Point(26) = {-c2, -c2+r2, 0, h};     // p9
Point(27) = {-c2, -c2-r2, 0, h};     // p9 


Circle(1) = {10, 1, 11};
Circle(2) = {11, 1, 10};

Circle(3) = {12, 2, 13};
Circle(4) = {13, 2, 12};

Circle(5) = {14, 3, 15};
Circle(6) = {15, 3, 14};

Circle(7) = {16, 4, 17};
Circle(8) = {17, 4, 16};

Circle(9) = {18, 5, 19};
Circle(10) = {19, 5, 18};

Circle(11) = {20, 6, 21};
Circle(12) = {21, 6, 20};

Circle(13) = {22, 7, 23};
Circle(14) = {23, 7, 22};

Circle(15) = {24, 8, 25};
Circle(16) = {25, 8, 24};

Circle(17) = {26, 9, 27};
Circle(18) = {27, 9, 26};

Line Loop(19) = {1,2};
Line Loop(20) = {3,4};
Line Loop(21) = {5,6};
Line Loop(22) = {7,8};
Line Loop(23) = {9,10};
Line Loop(24) = {11,12};
Line Loop(25) = {13,14};
Line Loop(26) = {15,16};
Line Loop(27) = {17,18};

Plane Surface(28) = {19, 20, 21, 22, 23, 24, 25, 26, 27};
Physical Surface(1) = {28};
Physical Line(1007) = {1, 2};
Physical Line(1001) = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};


// Settings
// This value gives the global element size factor (lower -> finer mesh)
Mesh.CharacteristicLengthFactor = 1.0 * 2^(-3);
// Insist on quads instead of default triangles
Mesh.RecombineAll = 1;
Mesh.RecombinationAlgorithm = 2;
// Violet instead of green base color for better visibility
Mesh.ColorCarousel = 0;
