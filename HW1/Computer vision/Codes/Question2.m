%{
The origin Ov of the car V is at a position (6,-8,1) in the world frame 
of reference W and rotated by 30 degree, Z-axis in W
%}

%{Translation matrix contains (-6,8,-1)%}
Trans_Ov = [1 0 0 -6;0 1 0 8; 0 0 1 -1; 0 0 0 1]

%{
Rotation about 30 degree, We take the -30 degree as rotation angle, as 
point rotates anti clockwise. 
%}
Rot_Ov = [ cosd(-30) -sind(-30) 0 0; sind(-30) cosd(-30) 0 0; 0 0 1 0; 0 0 0 1 ]

% Transformation Matrix %
vTw = Rot_Ov*Trans_Ov


%{
The origin Om of the mount reference is above 4 units vertically.
The axes are having tilt of 30 degree
%}

%{Translation matrix contains (0,0,-4)%}
Trans_Om = [1 0 0 0;0 1 0 0; 0 0 1 -4; 0 0 0 1]

%{
As shown in figure the rotation is 120 degree.
The new position z is now at 30 + 90 degree. The rotation of the point is 
anti clockwise about x axis.
%}
Rot_Om = [1 0 0 0;0 cosd(120) -sind(120) 0;0 sind(120) cosd(120) 0; 0 0 0 1]

% Transformation Matrix %
mTv = Rot_Om*Trans_Om



%{
The camera is placed along the negative Y at a distance 2 units
%}

%{Translation matrix contains (0,2,0)%}
Trans_Oc = [1 0 0 0;0 1 0 2; 0 0 1 0; 0 0 0 1]

% Transformation Matrix %
cTm = Trans_Oc

%{
To obtain the  world to camera transformation
%}

wTc =  cTm * mTv * vTw

%{
Solve for the camera coordinate 
%}

B = [0;0;0;1]
x = linsolve(wTc, B)
x = [1 0 0 0; 0 1 0 0; 0 0 1 0]*x 
x
%{
Plot the cube(world frame of reference)
%}
R = [0.8660    0.5000       0  
    0.2500   -0.4330   -0.8660  
   -0.4330    0.7500   -0.5000];


cam = plotCamera('Location',x,'Orientation',R,'Opacity',1);
grid on
axis equal
axis manual
xlim([-10,10]);
ylim([-10,10]);
zlim([-10,10]);

%{
Plot the cube. (0 0 0)
%}
origin = [0 0 0]
ver = [1 1 0;
    0 1 0;
    0 1 1;
    1 1 1;
    0 0 1;
    1 0 1;
    1 0 0;
    0 0 0];
fac = [1 2 3 4;
    4 3 5 6;
    6 7 8 5;
    1 2 8 7;
    6 7 1 4;
    2 3 5 8];
cube = [ver(:,1)*1+origin(1),ver(:,2)*1+origin(2),ver(:,3)*1+origin(3)];
patch('Faces',fac,'Vertices',cube,'FaceColor','red');


%{
Normalized image coordinates. 
With camera intrinsic parameters, only one is taken. 
Also other can be taken.
Calculating the pixel coordinates by changing the Cam_coor values.
%}

Cam_coor1 = [1; 1; 0; 1]
Cam_coor2 = [0; 1; 0; 1]
Cam_coor3 = [0; 1; 1; 1]
Cam_coor4 = [1; 1; 1; 1]
Cam_coor5 = [0; 0; 1; 1]
Cam_coor6 = [1; 0; 1; 1]
Cam_coor7 = [1; 0; 0; 1]
Cam_coor8 = [0; 0; 0; 1]

Camera_matrix = [  1.17706721e+03   0.00000000e+00   2.33392933e+02;
   0.00000000e+00   1.90902769e+03   1.43050760e+02;
   0.00000000e+00   0.00000000e+00   1.00000000e+00]

Pers_proj_mat = [
    1 0 0 0;
    0 1 0 0;
    0 0 1 0 ]

Point_cor = Camera_matrix * Pers_proj_mat * wTc * Cam_coor2
Point_cor = Point_cor/Point_cor(3)




