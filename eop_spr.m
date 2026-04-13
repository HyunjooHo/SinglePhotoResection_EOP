%% Single photo resection with EOP refinement
clear; close all; clc;
format long g

%% Load input data
inputFile = 'data/spr_gcp_inputs.csv';
T = readtable(inputFile);

%% Arrange input data
D = table2cell(T);
gcp = unique(D(:,1));                 % Unique GCP identifiers
num = size(D,1);                      % Number of image observations

%% Convert pixel coordinates to centered metric image coordinates
imgSize = [7952 5304];                % [width, height] in pixels
onepx = 4.4e-6;                       % Pixel size in meters

for i = 1:num
    D{i,8} = (D{i,6} - imgSize(1)/2) * onepx;
    D{i,9} = (imgSize(2)/2 - D{i,7}) * onepx;
end

%% Define initial IOP values
f_init = 32.784e-3;                   % Focal length in meters
f = f_init / onepx;                   % Focal length in pixels

% Initial IOPs in meters: [xp, yp, c]
init_IOP = 1e-3 * [-0.094217 -0.033517 32.784];

%% Define or estimate initial EOP values
% Set this flag to true to estimate the initial EOP from GCP geometry.
% Set it to false to use manually provided initial EOP values.
estimateInitialEOP = true;

if estimateInitialEOP
    A1 = [];
    Y1 = [];

    for k = 1:num
        XA = D{k,2}; YA = D{k,3}; hA = D{k,4};
        xa = D{k,8}; ya = D{k,9};

        if k < num
            XB = D{k+1,2}; YB = D{k+1,3}; hB = D{k+1,4};
            xb = D{k+1,8}; yb = D{k+1,9};
        else
            XB = D{k-1,2}; YB = D{k-1,3}; hB = D{k-1,4};
            xb = D{k-1,8}; yb = D{k-1,9};
        end

        syms H
        ABsq1 = (XB - XA)^2 + (YB - YA)^2;
        ABsq2 = (xb / init_IOP(3) * (H - hB) - xa / init_IOP(3) * (H - hA))^2 + ...
                (yb / init_IOP(3) * (H - hB) - ya / init_IOP(3) * (H - hA))^2;

        Ho = solve(ABsq2 - ABsq1 == 0, H);
        H1 = double(Ho(2));

        Xarb = xa * (H1 - hA) / init_IOP(3);
        Yarb = ya * (H1 - hA) / init_IOP(3);

        A1 = [A1;
              Xarb, -Yarb, 1, 0;
              Yarb,  Xarb, 0, 1];

        Y1 = [Y1; XA; YA];
    end

    dX1 = inv(A1.' * A1) * A1.' * Y1;

    init_kap = atan(dX1(2) / dX1(1));
    if dX1(1) < 0
        if dX1(2) < 0
            init_kap = init_kap - pi;
        else
            init_kap = init_kap + pi;
        end
    end

    init_EOP = [dX1(3) dX1(4) H1 0 0 init_kap];
else
    % Manual initial EOP values:
    % [Xo, Yo, Zo, omega, phi, kappa]
    init_EOP = [ ...
        318029.483 ...
        4159891.973 ...
        190.724 ...
        deg2rad(-2.929729) ...
        deg2rad(-1.345676) ...
        deg2rad(16.791045) ...
    ];
end

initial_EOP_deg = [init_EOP(1:3), rad2deg(init_EOP(4:6))]

%% Define collinearity equations
syms Xo Yo Zo omg phi kap xp yp c X Y Z

r11 = cos(phi) * cos(kap);
r12 = -cos(phi) * sin(kap);
r13 = sin(phi);

r21 = cos(omg) * sin(kap) + sin(omg) * sin(phi) * cos(kap);
r22 = cos(omg) * cos(kap) - sin(omg) * sin(phi) * sin(kap);
r23 = -sin(omg) * cos(phi);

r31 = sin(omg) * sin(kap) - cos(omg) * sin(phi) * cos(kap);
r32 = sin(omg) * cos(kap) + cos(omg) * sin(phi) * sin(kap);
r33 = cos(omg) * cos(phi);

Nx = r11 * (X - Xo) + r21 * (Y - Yo) + r31 * (Z - Zo);
Ny = r12 * (X - Xo) + r22 * (Y - Yo) + r32 * (Z - Zo);
De = r13 * (X - Xo) + r23 * (Y - Yo) + r33 * (Z - Zo);

x = xp - c * Nx / De;
y = yp - c * Ny / De;

Tay = jacobian([x y], [Xo Yo Zo omg phi kap]);

%% Perform iterative least-squares adjustment
Lim = 1e-6 * ones(6,1);
m = 1;
it_EOP = init_EOP;

while true
    A2 = zeros(num * 2, 6);
    Y2 = [];

    for k = 1:num
        Tv = double(subs(Tay, ...
            [Xo Yo Zo omg phi kap xp yp c X Y Z], ...
            [it_EOP, init_IOP, D{k,2:4}]));

        Iv = double(subs([x y], ...
            [Xo Yo Zo omg phi kap xp yp c X Y Z], ...
            [it_EOP, init_IOP, D{k,2:4}]));

        row1 = 2 * k - 1;
        row2 = 2 * k;

        A2(row1:row2, :) = Tv(:,1:6);
        Y2 = [Y2;
              D{k,8} - Iv(1);
              D{k,9} - Iv(2)];
    end

    dX = inv(A2.' * A2) * A2.' * Y2;

    if all(abs(dX) < Lim)
        break
    end

    it_EOP = it_EOP + dX.';
    m = m + 1;
end

%% Compute final statistics
iteration_num = m;
e = Y2 - A2 * dX;

final_EOP_deg = [it_EOP(1:3), rad2deg(it_EOP(4:6))];
variance_factor = (e.' * e) / (num * 2 - 6);
observation_minus_computed = Y2;

%% Display results
disp('Final EOP in degrees:')
disp(array2table(final_EOP_deg, ...
    'VariableNames', {'Xo','Yo','Zo','Omega_deg','Phi_deg','Kappa_deg'}))

disp('Variance factor:')
disp(variance_factor)

disp('Iteration count:')
disp(iteration_num)