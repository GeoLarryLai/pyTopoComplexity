%% Mexican Hat Continuous Wavelet Transform analysis
%% Originated from by Adam M. Booth
%% Optimized by Larry Syu-Heng Lai

% Initial Setup
clear; close all; clc;
baseDir = '/Users/larryslai/Library/CloudStorage/Dropbox/QGIS/WA LiDAR/';
inputDir = fullfile(baseDir, 'cropped sw WA DEM.tif');
outputDir = fullfile(baseDir, 'cropped_swWA_mexhat.tif');

% Read the DEM file once
%[dem, R] = geotiffread(inputDir);
[dem, R] = readgeoraster(inputDir);

% Wavelet Transform parameters (Change 'a' to vary scale (width) of wavelet)
dx = 1.8288; % Grid spacing
a = 4.1; % Scale of the wavelet

% Compute Mexican Hat Continuous Wavelet Transform
[C2] = conv2_mexh_lshl(dem, a, dx);
mexhat_result = abs(C2);

%% Display the result using imagesc
figure; % Create a new figure window
imagesc(mexhat_result); % Display the matrix as an image with scaled colors
colormap('jet'); % Apply a colormap (optional, 'jet' is just an example)
colorbar; % Show a colorbar indicating the scale of values
axis equal; % Set the axes aspect ratio so that the pixels are square
axis tight; % Fit the axes limits tightly around the data
title('Mexican Hat Wavelet Transform Result'); % Title for the plot

%% Write result to GeoTIFF
coordRefSysCode = 32149; %NAD83_Wash South
%coordRefSysCode = 32610; %WGS84_UTM Zone 10N
%coordRefSysCode = 4326 %WGS84

geotiffwrite(outputDir, mexhat_result, R, 'TiffType', 'bigtiff', ...
    'CoordRefSysCode', coordRefSysCode);
