import os
import numpy as np
import rasterio
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from scipy.signal import fftconvolve, convolve2d
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

class CWTMexHat:
    """
    A class for performing 2D Continuous Wavelet Transform (CWT) analysis using a Mexican Hat wavelet.

    This class implements the 2D-CWT method to measure topographic complexity (surface roughness)
    of a land surface from a Digital Elevation Model (DEM). The method quantifies the wavelet-based 
    curvature of the surface, which has been proposed as an effective geomorphic metric for 
    identifying and estimating the ages of historical deep-seated landslide deposits.

    Required parameters:
    -----------
    Lambda : float
        The wavelength (in meters) for the Mexican Hat wavelet.
    input_dir : str
        Path and filename of the input DEM file.
    output_dir : str
        Path and filename to save the output GeoTIFF file.

    Attributes:
    -----------
    Lambda : float
        The wavelength (in meters) for the Mexican Hat wavelet.
    ft2mUS : float
        Conversion factor from US survey feet to meters.
    ft2mInt : float
        Conversion factor from international feet to meters.
    input_dir : str
        Path to the input DEM file.
    Z : numpy.ndarray
        Array storing the input DEM data.
    result : numpy.ndarray
        Array storing the result of the 2D-CWT analysis.
    meta : dict
        Metadata of the input raster.
    conv_method : str
        Convolution method to use ('fft' or 'conv').
    chunk_processing : bool
        Whether to use Dask for chunk processing.
    chunksize : tuple
        Size of chunks for Dask processing.

    Methods:
    --------
    analyze(input_dir)
        Perform the 2D-CWT analysis on the input DEM.
    export_result(output_dir)
        Export the result of the analysis to a GeoTIFF file.
    plot_result()
        Plot the original DEM and the 2D-CWT result side by side.

    Example:
    --------
    >>> cwt = CWTMexHat(Lambda=15)
    >>> Z, result = cwt.analyze('input_dem.tif')
    >>> cwt.export_result('output_cwt.tif')
    >>> cwt.plot_result()

    References:
    -----------
    Booth, A.M., Roering, J.J., Perron, J.T., 2009. Automated landslide mapping using spectral 
    analysis and high-resolution topographic data: Puget Sound lowlands, Washington, and Portland 
    Hills, Oregon. Geomorphology 109, 132-147. https://doi.org/10.1016/j.geomorph.2009.02.027

    Torrence, C., Compo, G.P., 1998. A practical guide to wavelet analysis. Bulletin of the 
    American Meteorological Society 79 (1), 61–78.
    """

    def __init__(self, Lambda):
        self.Lambda = Lambda
        self.ft2mUS = 1200/3937  # US survey foot to meter conversion factor
        self.ft2mInt = 0.3048    # International foot to meter conversion factor
        self.input_dir = None
        self.Z = None
        self.result = None
        self.meta = None

    def conv2_mexh(self, Z, s, Delta):
        """
        Perform 2D convolution with a Mexican Hat wavelet.

        Parameters:
        -----------
        Z : numpy.ndarray
            Input elevation data.
        s : float
            Scale parameter for the wavelet.
        Delta : float
            Grid spacing of the input data.

        Returns:
        --------
        C : numpy.ndarray
            Result of the convolution.
        """
        X, Y = np.meshgrid(np.arange(-8 * s, 8 * s + 1), np.arange(-8 * s, 8 * s + 1))
        psi = (-1/(np.pi*(s * Delta)**4)) * (1 - (X**2 + Y**2)/(2 * s**2)) * np.exp(-(X**2 + Y**2)/(2* s**2))
        
        if self.conv_method == 'fft':
            C = (Delta**2) * fftconvolve(Z, psi, mode='same')
        elif self.conv_method == 'conv':
            C = (Delta**2) * convolve2d(Z, psi, mode='same')
        else:
            raise ValueError("Convolution method must be 'fft' or 'conv'.")
        
        return C

    def Delta_s_Calculate(self, input_dir):
        """
        1. Dxtract grid spacing (Delta) from the input DEM.
        2. Calculate scale parameter (s) based on Delta and the user-defined Lambda

        Parameters:
        -----------
        input_dir : str
            Path to the input DEM file.

        Returns:
        --------
        Delta : float
            Grid spacing of the input data.
        s : float
            Scale parameter for the wavelet.
        """
        with rasterio.open(input_dir) as src:
            gridsize = src.transform
            Zunit = src.crs.linear_units

        if any(unit in Zunit.lower() for unit in ["metre", "meter"]):
            Delta = np.mean([gridsize[0], -gridsize[4]])
        elif any(unit in Zunit.lower() for unit in ["foot", "feet", "ft"]):
            if any(unit in Zunit.lower() for unit in ["us", "united states"]):
                Delta = np.mean([gridsize[0] * self.ft2mUS, -gridsize[4] * self.ft2mUS])
            else:
                Delta = np.mean([gridsize[0] * self.ft2mInt, -gridsize[4] * self.ft2mInt])
        else:
            raise ValueError("The units of XY directions must be in feet or meters.")

        s = (self.Lambda/Delta)*((5/2)**(1/2)/(2*np.pi))

        return Delta, s

    def process_with_dask(self, input_dir, s, Delta):
        """
        Process the input DEM using Dask for parallel computation.

        Parameters:
        -----------
        input_dir : str
            Path to the input DEM file.
        s : float
            Scale parameter for the wavelet.
        Delta : float
            Grid spacing of the input data.
        """
        with rasterio.open(input_dir) as src:
            self.meta = src.meta.copy()
            Zunit = src.crs.linear_units
            self.Z = src.read(1)

        if any(unit in Zunit.lower() for unit in ["metre", "meter", "meters"]):
            pass
        elif any(unit in Zunit.lower() for unit in ["foot", "feet", "ft"]):
            if any(unit in Zunit.lower() for unit in ["us", "united states"]):
                self.Z = self.Z * self.ft2mUS
            else:
                self.Z = self.Z * self.ft2mInt
        else:
            raise ValueError("The unit of elevation 'z' must be in feet or meters.")

        dask_Z = da.from_array(self.Z, chunks=self.chunksize)

        processed_data = dask_Z.map_overlap(
            lambda block: np.abs(self.conv2_mexh(block, s, Delta)),
            depth=int(s * 4),
            boundary='reflect',
            trim=True,
            dtype=np.float32
        )

        with ProgressBar():
            self.result = processed_data.compute()
        
        fringeval = int(np.ceil(s * 2))
        self.result[:fringeval, :] = np.nan
        self.result[:, :fringeval] = np.nan
        self.result[-fringeval:, :] = np.nan
        self.result[:, -fringeval:] = np.nan

    def process_mexhat(self, input_dir, s, Delta):
        """
        Process the input DEM without using Dask (for smaller datasets).

        Parameters:
        -----------
        input_dir : str
            Path to the input DEM file.
        s : float
            Scale parameter for the wavelet.
        Delta : float
            Grid spacing of the input data.
        """
        with rasterio.open(input_dir) as src:
            self.Z = src.read(1)
            Zunit = src.crs.linear_units
            self.meta = src.meta.copy()

        if any(unit in Zunit.lower() for unit in ["metre", "meter", "meters"]):
            pass
        elif any(unit in Zunit.lower() for unit in ["foot", "feet", "ft"]):
            if any(unit in Zunit.lower() for unit in ["us", "united states"]):
                self.Z = self.Z * self.ft2mUS
            else:
                self.Z = self.Z * self.ft2mInt
        else:
            raise ValueError("The unit of elevation 'z' must be in feet or meters.")
        
        # Create a dask array from Z
        dask_Z = da.from_array(self.Z, chunks=self.Z.shape)
        
        # Create a delayed version of conv2_mexh
        delayed_conv2_mexh = dask.delayed(self.conv2_mexh)
        
        # Apply the convolution
        C2 = delayed_conv2_mexh(dask_Z, s, Delta)
        
        # Compute the absolute value
        result = da.abs(C2)
        
        # Compute the result with a progress bar
        with ProgressBar():
            self.result = result.compute()

        # Replace edges with NaN
        cropedge = np.ceil(s * 4)
        fringeval = int(cropedge)
        self.result[:fringeval, :] = np.nan
        self.result[:, :fringeval] = np.nan
        self.result[-fringeval:, :] = np.nan
        self.result[:, -fringeval:] = np.nan

    def analyze(self, input_dir, conv_method='fft', chunk_processing=True, chunksize=(512, 512)):
        """
        Perform the 2D-CWT analysis on the input DEM.

        Parameters:
        -----------
        input_dir : str
            Path to the input DEM file.
        conv_method : str, optional
            Convolution method to use. Either 'fft' (default) or 'conv'.
        chunk_processing : bool, optional
            Whether to use Dask for chunk processing (default is True).
        chunksize : tuple, optional
            Size of chunks for Dask processing (default is (512, 512)).

        Returns:
        --------
        Z : numpy.ndarray
            The input elevation data.
        result : numpy.ndarray
            The result of the 2D-CWT analysis.
        meta : dict
            Metadata of the input raster.
        """
        self.input_dir = input_dir
        self.conv_method = conv_method
        self.chunk_processing = chunk_processing
        self.chunksize = chunksize
        
        Delta, s = self.Delta_s_Calculate(input_dir)
        
        if self.chunk_processing:
            self.process_with_dask(input_dir, s, Delta)
        else:
            self.process_mexhat(input_dir, s, Delta)
        
        return self.Z, self.result

    def export_result(self, output_dir):
        """
        Export the result of the analysis to a GeoTIFF file.

        Parameters:
        -----------
        output_dir : str
            Path where the output GeoTIFF will be saved.
        """
        if self.meta is None:
            raise ValueError("Metadata is missing. Cannot export the result.")
        
        self.meta.update(dtype=rasterio.float32, count=1, compress='deflate', bigtiff='IF_SAFER')
        
        with rasterio.open(output_dir, 'w', **self.meta) as dst:
            dst.write(self.result.astype(rasterio.float32), 1)
        
        print(f"'{os.path.basename(output_dir)}' is saved")

    def plot_result(self, output_dir=None, savefig=True, figshow=True, showhillshade=True, cwtcolormax=None):
        """
        Plot the original DEM and the 2D-CWT result side by side, or only the 2D-CWT result.

        Parameters:
        -----------
        output_dir : str, optional
            Specified directory to save the figure. If None, uses the input file's directory.
        savefig : bool, optional
            Whether to save the figure as a PNG file (default is True).
        figshow : bool, optional
            Whether to display the figure (default is True).
        showhillshade : bool, optional
            Whether to show the hillshade plot alongside the roughness data (default is True).
        cwtcolormax : float, optional
            Maximum value for roughness color scale. If None, uses data-derived values.
        """
        if self.Z is None or self.result is None or self.input_dir is None:
            raise ValueError("Analysis must be run before plotting results.")

        input_file = os.path.basename(self.input_dir)
        base_dir = output_dir if output_dir else os.path.dirname(self.input_dir)

        with rasterio.open(self.input_dir) as src:
            gridsize = src.transform
            Zunit = src.crs.linear_units

        if showhillshade:
            # Scenario with hillshade
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
            
            # Plot the hillshade
            ls = LightSource(azdeg=315, altdeg=45)
            hs = axes[0].imshow(ls.hillshade(self.Z, vert_exag=2), cmap='gray')
            axes[0].set_title(input_file)
            axes[0].set_xlabel(f'X-axis grids \n(grid size ≈ {round(gridsize[0],4)} [{Zunit}])')
            axes[0].set_ylabel(f'Y-axis grids \n(grid size ≈ {-round(gridsize[4],4)} [{Zunit}])')
            cbar1 = fig.colorbar(hs, ax=axes[0], orientation='horizontal', fraction=0.045, pad=0.13)
            cbar1.ax.set_visible(False)

            # Plot the 2D-CWT roughness
            im = axes[1].imshow(self.result, cmap='viridis')
            if cwtcolormax is None:
                im.set_clim(0, round(np.nanpercentile(self.result, 99), 2))
            else:
                im.set_clim(0, cwtcolormax)
            axes[1].set_title(f'2D-CWT surface roughness [m$^{{-1}}$] \n measured with {self.Lambda}m Mexican Hat wavelet')
            axes[1].set_xlabel(f'X-axis grids \n(grid size ≈ {round(gridsize[0],4)} [{Zunit}])')
            axes[1].set_ylabel(f'Y-axis grids \n(grid size ≈ {-round(gridsize[4],4)} [{Zunit}])')
            cbar2 = fig.colorbar(im, ax=axes[1], orientation='horizontal', fraction=0.045, pad=0.13)
        else:
            # Scenario without hillshade
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
            
            # Plot only the 2D-CWT roughness
            im = ax.imshow(self.result, cmap='viridis')
            if cwtcolormax is None:
                im.set_clim(0, round(np.nanpercentile(self.result, 99), 2))
            else:
                im.set_clim(0, cwtcolormax)
            ax.set_title(f'2D-CWT surface roughness [m$^{{-1}}$] \n measured with {self.Lambda}m Mexican Hat wavelet')
            ax.set_xlabel(f'X-axis grids \n(grid size ≈ {round(gridsize[0],4)} [{Zunit}])')
            ax.set_ylabel(f'Y-axis grids \n(grid size ≈ {-round(gridsize[4],4)} [{Zunit}])')
            cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.045, pad=0.13)

        plt.tight_layout()
        
        if savefig:
            output_filename = os.path.splitext(input_file)[0] + f'_pyCWTMexHat({self.Lambda}m).png'
            output_path = os.path.join(base_dir, output_filename)
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            print(f"Figure saved as '{os.path.basename(output_path)}'")

        if figshow:
            plt.show()
        else:
            plt.close(fig)