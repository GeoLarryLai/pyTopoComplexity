import os
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from tqdm.auto import tqdm
import dask.array as da
from dask.diagnostics import ProgressBar

class TPI:
    """
    A class for calculating the Terrain Position Index (TPI) of a land surface using Digital Elevation Model (DEM) data.

    The Terrain Position Index (TPI) measures the topographic position of a point relative to the surrounding terrain. 
    It determines whether a point is situated on a topographic high, low, or a slope. TPI is calculated by comparing 
    the elevation of a cell to the mean elevation of its surrounding cells within a specified neighborhood.

    In terrestrial research, TPI is commonly referred to as the ‘Topographic Position Index’ (Weiss, 2001) In oceanography and 
    marine ecology, it is known as the ‘Bathymetric Position Index (BPI)’ (Wilson et al., 2007).

    Parameters:
    -----------
    window_size : int, optional
        The size of the moving window for TPI calculation. Must be an odd integer >= 3. Default is 3.

    Attributes:
    -----------
    ft2mUS : float
        Conversion factor from US survey feet to meters.
    ft2mInt : float
        Conversion factor from international feet to meters.
    window_size : int
        Size of the moving window for TPI calculation.
    Z : numpy.ndarray
        The input elevation data.
    TPI : numpy.ndarray
        The calculated Terrain Position Index.
    TPIabs : numpy.ndarray
        The absolute values of the Terrain Position Index.
    input_dir : str
        Path to the input DEM file.
    window_size_m : float
        Window size in meters.
    meta : dict
        Metadata of the input raster.

    Methods:
    --------
    analyze(input_dir, chunk_processing=True, chunksize=(512, 512))
        Perform TPI analysis on the input DEM.
    export_result(output_dir, output_abs_dir)
        Export the TPI and absolute TPI results to GeoTIFF files.
    plot_result(savefig=True)
        Plot and optionally save the original DEM, TPI, and absolute TPI results.

    Example:
    --------
    >>> tpi = TPI(window_size=11)
    >>> Z, TPI, TPIabs, meta, window_m = tpi.analyze('input_dem.tif')
    >>> tpi.export_result('output_tpi.tif', 'output_tpi_abs.tif')
    >>> tpi.plot_result()

    References:
    -----------
    Weiss, A. D. 2001. Topographic Positions and Landforms Analysis (poster),
    ESRI International User Conference, July 2001. San Diego, CA: ESRI.
    
    Wilson, M.F.J., O’Connell, B., Brown, C., Guinan, J.C., Grehan, A.J., 2007. 
    Multiscale Terrain Analysis of Multibeam Bathymetry Data for Habitat Mapping on 
    the Continental Slope. Marine Geodesy 30, 3-35. https://doi.org/10.1080/01490410701295962.
    """

    def __init__(self, window_size=10):
        self.ft2mUS = 1200/3937  # US survey foot to meter conversion factor
        self.ft2mInt = 0.3048    # International foot to meter conversion factor
        self.window_size = window_size
        self.Z = None
        self.TPI = None
        self.TPIabs = None
        self.input_dir = None
        self.window_size_m = None
        self.meta = None
        self.cell_size = None

    def calculate_TPI(self, Z):
        """
        Calculate the Terrain Position Index (TPI) for a digital elevation model (DEM).
        
        Parameters:
        -----------
        Z : numpy.ndarray or dask.array.Array
            The input elevation data.
        
        Returns:
        --------
        TPI : numpy.ndarray or dask.array.Array
            The calculated Terrain Position Index.
        """
        # Calculate the mean elevation within the window
        mean_Z = uniform_filter(Z, size=self.window_size)
        
        # TPI is the difference between the cell elevation and the mean elevation
        TPI = Z - mean_Z

        return TPI

    def _process_non_chunked(self, Z):
        """
        Process the entire DEM without chunking, using a tqdm progress bar.

        Parameters:
        -----------
        Z : numpy.ndarray
            The input elevation data.

        Returns:
        --------
        TPI : numpy.ndarray
            The calculated Terrain Position Index.
        TPIabs : numpy.ndarray
            The absolute values of the Terrain Position Index.
        """
        height, width = Z.shape
        TPI = np.zeros_like(Z)
        TPIabs = np.zeros_like(Z)

        # Use tqdm to create a progress bar
        for i in tqdm(range(height), desc="Processing rows", unit="row"):
            # Calculate TPI for the entire row at once
            mean_Z = uniform_filter(Z[max(0, i-self.window_size//2):min(i+self.window_size//2+1, height), 
                                      max(0, -self.window_size//2):min(width+self.window_size//2, width)], 
                                    size=self.window_size)
            TPI[i] = Z[i] - mean_Z[min(self.window_size//2, i):min(self.window_size//2+1, i+1)]

        # Calculate absolute TPI values
        TPIabs = np.absolute(TPI)

        return TPI, TPIabs

    def _process_chunked(self, Z, height, width):
        """
        Process the DEM in chunks using Dask for improved memory efficiency and parallelization.

        Parameters:
        -----------
        Z : numpy.ndarray
            The input elevation data.
        height : int
            Height of the DEM.
        width : int
            Width of the DEM.

        Returns:
        --------
        TPI : numpy.ndarray
            The calculated Terrain Position Index.
        TPIabs : numpy.ndarray
            The absolute values of the Terrain Position Index.
        """
        # Create a dask array from Z
        dask_Z = da.from_array(Z, chunks=self.chunksize)
        
        # Calculate TPI using dask
        dask_TPI = dask_Z.map_overlap(
            self.calculate_TPI,
            depth=self.window_size//2,
            boundary='reflect'
        )
        
        # Calculate absolute TPI
        dask_TPIabs = abs(dask_TPI)
        
        # Compute the result with a progress bar
        with ProgressBar():
            TPI = dask_TPI.compute()

        # Calculate absolute TPI values
        TPIabs = np.absolute(TPI)

        return TPI, TPIabs

    def analyze(self, input_dir, chunk_processing=True, chunksize=(512, 512)):
        """
        Perform TPI analysis on the input DEM.

        Parameters:
        -----------
        input_dir : str
            Path to the input DEM file.
        chunk_processing : bool, optional
            Whether to use chunk processing for large DEMs. Default is True.
        chunksize : tuple of int, optional
            Size of chunks for processing. Default is (512, 512).

        Returns:
        --------
        Z : numpy.ndarray
            The input elevation data.
        TPI : numpy.ndarray
            The calculated Terrain Position Index.
        TPIabs : numpy.ndarray
            The absolute values of the Terrain Position Index.
        meta : dict
            Metadata of the input raster.
        window_size_m : float
            Window size in meters.
        """
        self.input_dir = input_dir
        self.chunksize = chunksize

        with rasterio.open(input_dir) as src:
            self.meta = src.meta.copy()
            Z = src.read(1)
            self.cell_size = src.transform[0]
            Zunit = src.crs.linear_units

        # Convert to meters if necessary
        if any(unit in Zunit.lower() for unit in ["metre", "meter", "meters"]):
            pass
        elif any(unit in Zunit.lower() for unit in ["foot", "feet", "ft"]):
            if any(unit in Zunit.lower() for unit in ["us", "united states"]):
                Z = Z * self.ft2mUS
                self.cell_size = self.cell_size * self.ft2mUS
            else:
                Z = Z * self.ft2mInt
                self.cell_size = self.cell_size * self.ft2mInt
        else:
            raise ValueError("The unit of elevation 'z' must be in feet or meters.")

        # Calculate window size in meters
        self.window_size_m = self.window_size * self.cell_size

        if chunk_processing:
            self.TPI, self.TPIabs = self._process_chunked(Z, Z.shape[0], Z.shape[1])
        else:
            self.TPI, self.TPIabs = self._process_non_chunked(Z)

        # Mask edge with NaN (no data) values to remove artifacts
        fringeval = self.window_size // 2
        self.TPI[:fringeval, :] = np.nan
        self.TPI[:, :fringeval] = np.nan
        self.TPI[-fringeval:, :] = np.nan
        self.TPI[:, -fringeval:] = np.nan
        self.TPIabs[:fringeval, :] = np.nan
        self.TPIabs[:, :fringeval] = np.nan
        self.TPIabs[-fringeval:, :] = np.nan
        self.TPIabs[:, -fringeval:] = np.nan

        self.Z = Z

        return self.Z, self.TPI, self.TPIabs, self.window_size_m

    def export_result(self, output_dir, output_abs_dir):
        """
        Export the TPI and absolute TPI results to GeoTIFF files.

        Parameters:
        -----------
        output_dir : str
            Path to save the TPI result.
        output_abs_dir : str
            Path to save the absolute TPI result.

        Raises:
        -------
        ValueError
            If the metadata is missing and results cannot be exported.
        """
        if self.meta is None:
            raise ValueError("Metadata is missing. Cannot export the result.")
        
        # Update metadata for output GeoTIFF
        self.meta.update(dtype=rasterio.float32, compress='deflate', tiled=True, bigtiff='IF_SAFER')

        # Write TPI to a new GeoTIFF
        with rasterio.open(output_dir, 'w', **self.meta) as dst:
            dst.write(self.TPI.astype(rasterio.float32), 1)
        print(f"Processed TPI result saved to {os.path.basename(output_dir)}")

        # Write TPIabs to a new GeoTIFF
        with rasterio.open(output_abs_dir, 'w', **self.meta) as dst:
            dst.write(self.TPIabs.astype(rasterio.float32), 1)
        print(f"Processed absolute TPI result saved to {os.path.basename(output_abs_dir)}")

    def plot_result(self, savefig=True):
        """
        Plot the original DEM, TPI, and absolute TPI results side by side.

        Parameters:
        -----------
        savefig : bool, optional
            Whether to save the figure as a PNG file. Default is True.

        Raises:
        -------
        ValueError
            If the analysis hasn't been run before calling this method.
        """
        if self.Z is None or self.TPI is None or self.TPIabs is None:
            raise ValueError("Analysis must be run before plotting results.")

        input_file = os.path.basename(self.input_dir)
        base_dir = os.path.dirname(self.input_dir)

        with rasterio.open(self.input_dir) as src:
            transform = src.transform
            crs = src.crs

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

        # Plot the hillshade
        ls = LightSource(azdeg=315, altdeg=45)
        hs = axes[0].imshow(ls.hillshade(self.Z, vert_exag=2), cmap='gray')
        axes[0].set_title(input_file)
        axes[0].set_xlabel(f'X-axis grids \n(grid size ≈ {round(transform[0],4)} [{crs.linear_units}])')
        axes[0].set_ylabel(f'Y-axis grids \n(grid size ≈ {-round(transform[4],4)} [{crs.linear_units}])')
        cbar1 = fig.colorbar(hs, ax=axes[0], orientation='horizontal', fraction=0.045, pad=0.13)
        cbar1.ax.set_visible(False)

        # Plot the TPI
        im = axes[1].imshow(self.TPI, cmap='RdBu_r')
        boundary = np.max([np.abs(np.nanpercentile(self.TPI, 1)), np.abs(np.nanpercentile(self.TPI, 99))])
        im.set_clim(-round(boundary,2), round(boundary,2))
        axes[1].set_title(f'Terrain Position Index [m]\n(~{round(self.window_size_m, 2)}m x ~{round(self.window_size_m, 2)}m window)')
        axes[1].set_xlabel(f'X-axis grids \n(grid size ≈ {round(transform[0],4)} [{crs.linear_units}])')
        axes[1].set_ylabel(f'Y-axis grids \n(grid size ≈ {-round(transform[4],4)} [{crs.linear_units}])')
        cbar2 = fig.colorbar(im, ax=axes[1], orientation='horizontal', fraction=0.045, pad=0.13)

        # Plot the TPIabs
        im = axes[2].imshow(self.TPIabs, cmap='viridis')
        im.set_clim(round(np.nanpercentile(self.TPIabs, 1), 2), round(np.nanpercentile(self.TPIabs, 99), 2))
        axes[2].set_title(f'Absolute Values of Terrain Position Index [m]\n(~{round(self.window_size_m, 2)}m x ~{round(self.window_size_m, 2)}m window)')
        axes[2].set_xlabel(f'X-axis grids \n(grid size ≈ {round(transform[0],4)} [{crs.linear_units}])')
        axes[2].set_ylabel(f'Y-axis grids \n(grid size ≈ {-round(transform[4],4)} [{crs.linear_units}])')
        cbar3 = fig.colorbar(im, ax=axes[2], orientation='horizontal', fraction=0.045, pad=0.13)

        plt.tight_layout()

        if savefig:
            output_filename = os.path.splitext(input_file)[0] + f'_pyTPI({round(self.window_size_m, 2)}m).png'
            output_dir = os.path.join(base_dir, output_filename)
            plt.savefig(output_dir, dpi=200, bbox_inches='tight')
            print(f"Figure saved as '{output_filename}'")

        plt.show()