function [C, frq, wave] = conv2_mexh_lshl(dem, a, dx)

    %start timer
    tic     

    % Check inputs
    if nargin ~= 3
        error('conv2_mexh_new requires exactly three input arguments.');
    end
    if ~isnumeric(dem) || ~isnumeric(a) || ~isnumeric(dx)
        error('Input arguments dem, a, and dx must be numeric.');
    end

    % Generate the Mexican Hat wavelet kernel at wavelet scale a.
    sz = ceil(8 * a); % Kernel size, assuming the wavelet decays to 0 at the edges
    [X, Y] = meshgrid(-sz:sz, -sz:sz);

    % Scaled Mexican Hat wavelet (psi); units of [1/(m^4)]
    psi = (-1/(pi*(a*dx)^4)) * (1 - (X.^2 + Y.^2) / (2*a^2)) .* exp(-(X.^2 + Y.^2) / (2*a^2));

    % Convolve dem with psi; units of [(m^2) x (m) x (1/(m^4)) = (1/m)]
    C = (dx^2) * conv2(dem * 0.3048, psi, 'same');

    % Mask edge effects with NaN values.
    nrows = size(C, 1);
    ncols = size(C, 2);
    fringeval = ceil(a*4);
    C([1:fringeval, end-fringeval+1:end], :) = NaN;
    C(:, [1:fringeval, end-fringeval+1:end]) = NaN;

    % Frequency and wavelength calculations
    wave = 2*pi*dx*a / sqrt(5/2); % Wavelength
    frq = 1 / wave; % Frequency

    %stop timer
    toc     
end
