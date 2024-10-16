import os
import numpy as np
import rasterio
import dask.array as da
from dask.diagnostics import ProgressBar
from numba import jit
from scipy.ndimage import generic_filter
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

class RugosityIndex:
    """
    A class for calculating the rugosity index of a land surface using Digital Elevation Model (DEM) data.

    The rugosity index is a measure of terrain complexity, calculated as the ratio of the real surface area
    to the planar area. This implementation uses a moving window approach and adapts the triangulated
    irregular networks method described by Jenness (2004). 
    
    If no local slope correction is applied, the planimetric area is considered to be the horizontal planar 
    area of the moving window, as decribed in Jenness (2004). Another approach considers slope correction 
    where to the planimetric area is projected onto an plane of the local gradient (Du Preez, 2015).

    Required parameters:
    -----------
    window_size : int
        The size of the moving window for rugosity calculation. Must be an odd integer >= 3.
    input_dir : str
        Path and filename of the input DEM file.
    output_dir : str
        Path and filename to save the output GeoTIFF file.

    Attributes:
    -----------
    window_size : int
        Size of the moving window for rugosity calculation.
    ft2mUS : float
        Conversion factor from US survey feet to meters.
    ft2mInt : float
        Conversion factor from international feet to meters.
    input_dir : str
        Path to the input DEM file.
    Z : numpy.ndarray
        The input elevation data.
    result : numpy.ndarray
        The calculated rugosity index.
    meta : dict
        Metadata of the input raster.
    window_size_m : float
        Window size in meters.
    slope_correction : bool
        Whether slope correction is applied.
    chunk_processing : bool
        Whether to use Dask for chunk processing.
    chunksize : tuple
        Size of chunks for Dask processing.

    Methods:
    --------
    analyze(input_dir, slope_correction=True, chunk_processing=True, chunksize=(512, 512))
        Perform rugosity analysis on the input DEM.
    export_result(output_dir)
        Export the rugosity result to a GeoTIFF file.
    plot_result(output_dir=None, savefig=True, figshow=True, showhillshade=True)
        Plot and optionally save the original DEM and rugosity result.

    Example:
    --------
    >>> rgty = RugosityIndex(window_size=11)
    >>> Z, result, window_m = rgty.analyze('input_dem.tif')
    >>> rgty.export_result('output_rugosity.tif')
    >>> rgty.plot_result()

    References:
    -----------
    Jenness, J.S. (2004). Calculating landscape surface area from digital elevation models.
    Wildlife Society Bulletin, 32: 829-839. https://doi.org/10.2193/0091-7648(2004)032[0829:CLSAFD]2.0.CO;2

    Du Preez, C. (2015) A new arc–chord ratio (ACR) rugosity index for quantifying three-dimensional 
    landscape structural complexity. Landscape Ecol 30, 181–192. https://doi.org/10.1007/s10980-014-0118-8
    """

    def __init__(self, window_size):
        self.ft2mUS = 1200/3937  # US survey foot to meter conversion factor
        self.ft2mInt = 0.3048    # International foot to meter conversion factor
        self.window_size = window_size
        self.Z = None
        self.result = None
        self.meta = None
        self.input_dir = None
        self.window_size_m = None

    @staticmethod
    @jit(nopython=True)
    def compute_triangle_areas(values, cell_size):
        """
        Compute the surface area within a moving window using triangulated irregular networks.

        This method calculates the surface area by dividing the window into triangles,
        computing the area of each triangle using a robust implementation of Heron's formula,
        and summing the results.

        Parameters:
        -----------
        values : numpy.ndarray
            Elevation values within the moving window, flattened to 1D array.
        cell_size : float
            Size of each cell in the DEM (in the same units as values).

        Returns:
        --------
        float
            Total surface area within the window.

        Notes:
        ------
        This method uses numpy's hypot function for robust distance calculations,
        which helps prevent numerical instability with extreme elevation differences.
        It also includes safeguards against invalid area calculations.

        The surface area is approximated using 8 triangles per moving window, 
        connecting the center point with its 8 neighbors, as described by Jenness (2004).
        """
        window_size = int(np.sqrt(len(values)))
        center = values[len(values) // 2]
        total_surface_area = 0.0
        
        for i in range(window_size - 1):
            for j in range(window_size - 1):
                idx = i * window_size + j
                corner_dist = np.sqrt(2) * cell_size

                # Calculate edges using hypot for robustness
                orthogonal_edges = np.hypot(center - values[idx], cell_size) / 2
                diagonal_edges = np.hypot(center - values[idx], corner_dist) / 2
                adjacent_edges = np.hypot(values[idx] - values[idx + 1], cell_size) / 2
                
                # Calculate triangle areas using Heron's formula
                s = (orthogonal_edges + diagonal_edges + adjacent_edges) / 2
                area = np.sqrt(np.abs(s * (s - orthogonal_edges) * (s - diagonal_edges) * (s - adjacent_edges)))
                
                # Handle potential numerical instability
                if np.isnan(area) or area <= 0:
                    area = cell_size ** 2 / 2  # fallback to planar area
                
                total_surface_area += area * 8  # Eight triangles per square
        
        return total_surface_area

    @staticmethod
    @jit(nopython=True)
    def calculate_window_slope(values, cell_size):
        """
        Calculate the slope for a window using the planar method, adapted for variable window sizes.
        
        Parameters:
        -----------
        values : numpy.ndarray
            Elevation values within the moving window, flattened to 1D array.
        cell_size : float
            Size of each cell in the DEM (in the same units as values).
        
        Returns:
        --------
        float
            Slope of the window in degrees.

        The planar method for estimating the slope of center cell is adatped from 
        https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/how-slope-works.htm
        """
        window_size = int(np.sqrt(len(values)))
        
        # Extract the 9 key points
        a = values[0]                           # top-left corner
        b = values[window_size // 2]            # top-middle
        c = values[window_size - 1]             # top-right corner
        d = values[(window_size // 2) * window_size]  # middle-left
        e = values[len(values) // 2]            # center
        f = values[(window_size // 2 + 1) * window_size - 1]  # middle-right
        g = values[window_size * (window_size - 1)]  # bottom-left corner
        h = values[window_size * (window_size - 1) + window_size // 2]  # bottom-middle
        i = values[-1]                          # bottom-right corner
        
        # Calculate rates of change in x and y directions
        dz_dx = ((c + 2*f + i) - (a + 2*d + g)) / (8 * cell_size * (window_size - 1))
        dz_dy = ((g + 2*h + i) - (a + 2*b + c)) / (8 * cell_size * (window_size - 1))
        
        # Calculate slope
        rise_run = np.sqrt(dz_dx**2 + dz_dy**2)
        slope_radians = np.arctan(rise_run)
        
        return slope_radians

    def calculate_rugosity(self, Z, cell_size, slope_correction):
        """
        Calculate the rugosity index for the entire DEM.

        This method applies a moving window filter to the input DEM to compute
        the rugosity index for each cell. It uses either chunked processing with
        dask for large DEMs or a single-threaded approach for smaller ones.

        Parameters:
        -----------
        Z : numpy.ndarray
            Input elevation data.
        cell_size : float
            Size of each cell in the DEM (in the same units as Z).
        slope_correction : bool
            Whether to apply slope correction to the planar area calculation.

        Returns:
        --------
        numpy.ndarray
            Rugosity index for each cell in the DEM.

        Raises:
        -------
        ValueError
            If the window_size is not an odd integer >= 3.

        Notes:
        ------
        The rugosity index is calculated as the ratio of the real surface area
        to the planar area within each moving window. This method uses the
        compute_triangle_areas function to calculate the surface area.

        When slope correction is applied (default), it returns the arc-chord ratio (ACR) 
        rugosity index (Du Preez, 2015). When slope correction is turned off, it returns 
        the conventional rugosity index (Jenness, 2004).

        """
        if self.window_size % 2 == 0 or self.window_size < 3:
            raise ValueError("'window_size' must be an odd integer >= 3.")

        def rugosity_filter(values):
            surface_area = self.compute_triangle_areas(values, cell_size)
            flat_planar_area = ((self.window_size - 1) * cell_size)**2
            
            if slope_correction:
                slope_radians = self.calculate_window_slope(values, cell_size)
                planar_area = flat_planar_area / np.cos(slope_radians)
            else:
                planar_area = flat_planar_area
            
            rugosity = surface_area / planar_area
            return rugosity

        if self.chunk_processing:
            # Use dask for parallel processing with user-defined chunk size
            dask_Z = da.from_array(Z, chunks=self.chunksize)
            rugosity = dask_Z.map_overlap(
                lambda block: generic_filter(block, rugosity_filter, size=self.window_size, mode='nearest'),
                depth=self.window_size//2,
                boundary='reflect'
            )
            
            # Compute with progress bar
            with ProgressBar():
                result = rugosity.compute()
        else:
            # Process without chunking, using tqdm for progress
            total_pixels = Z.shape[0] * Z.shape[1]
            with tqdm(total=total_pixels, desc="Processing pixels") as pbar:
                def progress_filter(values):
                    pbar.update(1)
                    return rugosity_filter(values)
                result = generic_filter(Z, progress_filter, size=self.window_size, mode='nearest')
        
        return result

    def analyze(self, input_dir, slope_correction=True, chunk_processing=True, chunksize=(512, 512)):
        """
        Perform rugosity analysis on the input DEM.

        This method reads the input DEM, calculates the rugosity index for each pixel using
        the specified moving window size, and handles unit conversions if necessary.

        Parameters:
        -----------
        input_dir : str
            Path to the input DEM file.
        slope_correction : bool, optional
            Whether to apply slope correction to the planar area calculation. Default is True.
        chunk_processing : bool, optional
            Whether to use chunk processing for large DEMs. Default is True.
        chunksize : tuple of int, optional
            Size of chunks for processing. Default is (512, 512).

        Returns:
        --------
        Z : numpy.ndarray
            The input elevation data.
        result : numpy.ndarray
            The calculated rugosity index.
        meta : dict
            Metadata of the input raster.
        window_size_m : float
            Window size in meters.
        """
        self.input_dir = input_dir
        self.chunk_processing = chunk_processing
        self.chunksize = chunksize
        self.slope_correction = slope_correction

        with rasterio.open(input_dir) as src:
            self.meta = src.meta.copy()
            self.Z = src.read(1)
            Zunit = src.crs.linear_units
            gridsize = src.transform[0]
            
            # Convert to meters if necessary            
            if any(unit in Zunit.lower() for unit in ["metre", "meter", "meters"]):
                pass
            elif any(unit in Zunit.lower() for unit in ["foot", "feet", "ft"]):
                if any(unit in Zunit.lower() for unit in ["us", "united states"]):
                    self.Z = self.Z * self.ft2mUS
                    gridsize = gridsize * self.ft2mUS
                else:
                    self.Z = self.Z * self.ft2mInt
                    gridsize = gridsize * self.ft2mInt
            else:
                raise ValueError("The unit of elevation 'z' must be in feet or meters.")

            # Calculate rugosity
            self.result = self.calculate_rugosity(self.Z, gridsize, slope_correction)        

            # Replace edges with NaN
            fringeval = self.window_size // 2 + 1
            self.result[:fringeval, :] = np.nan
            self.result[:, :fringeval] = np.nan
            self.result[-fringeval:, :] = np.nan
            self.result[:, -fringeval:] = np.nan

            # Calculate window size in meters
            self.window_size_m = gridsize * self.window_size

        return self.Z, self.result, self.window_size_m

    def export_result(self, output_dir):
        """
        Export the rugosity result to a GeoTIFF file.

        This method saves the calculated rugosity index as a GeoTIFF file,
        preserving the georeferencing information from the input DEM.

        Parameters:
        -----------
        output_dir : str
            Path where the output GeoTIFF will be saved.

        Raises:
        -------
        ValueError
            If the metadata is missing and results cannot be exported.
        """
        if self.meta is None:
            raise ValueError("Metadata is missing. Cannot export the result.")
        
        # Setup output metadata
        self.meta.update(dtype=rasterio.float32, count=1, compress='deflate', bigtiff='IF_SAFER')
        
        # Write result to GeoTiff
        with rasterio.open(output_dir, 'w', **self.meta) as dst:
            dst.write(self.result.astype(rasterio.float32), 1)
        
        print(f"'{os.path.basename(output_dir)}' is saved")

    def plot_result(self, output_dir=None, savefig=True, figshow=True, showhillshade=True, rugositycolormax=None):
        """
        Plot the original DEM and the rugosity result side by side, or only the rugosity result.

        Parameters:
        -----------
        output_dir : str, optional
            Specified directory to save the figure. If None, uses the input file's directory.
        savefig : bool, optional
            Whether to save the figure as a PNG file (default is True).
        figshow : bool, optional
            Whether to display the figure (default is True).
        showhillshade : bool, optional
            Whether to show the hillshade plot alongside the rugosity data (default is True).
        rugositycolormax : float, optional
            Maximum value for rugosity color scale. If None, uses data-derived values.
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

            # Plot the rugosity
            if self.slope_correction:
                im = axes[1].imshow(self.result, cmap='viridis')
                if rugositycolormax is None:
                    im.set_clim(round(np.nanpercentile(self.result, 1), 2), round(np.nanpercentile(self.result, 99), 2))
                else:
                    im.set_clim(1, rugositycolormax)
                axes[1].set_title(f'Arc-Chord Ratio (ACR) Rugosity Index\n(~{round(self.window_size_m, 2)}m x ~{round(self.window_size_m, 2)}m window)')
            else:
                im = axes[1].imshow(self.result, cmap='plasma')
                if rugositycolormax is None:
                    im.set_clim(round(np.nanpercentile(self.result, 1), 2), round(np.nanpercentile(self.result, 99), 2))
                else:
                    im.set_clim(1, rugositycolormax)
                axes[1].set_title(f'Conventional Rugosity Index\n(~{round(self.window_size_m, 2)}m x ~{round(self.window_size_m, 2)}m window)')
            axes[1].set_xlabel(f'X-axis grids \n(grid size ≈ {round(gridsize[0],4)} [{Zunit}])')
            axes[1].set_ylabel(f'Y-axis grids \n(grid size ≈ {-round(gridsize[4],4)} [{Zunit}])')
            cbar2 = fig.colorbar(im, ax=axes[1], orientation='horizontal', fraction=0.045, pad=0.13)
        else:
            # Scenario without hillshade
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
            
            # Plot only the rugosity
            if self.slope_correction:
                im = ax.imshow(self.result, cmap='viridis')
                if rugositycolormax is None:
                    im.set_clim(round(np.nanpercentile(self.result, 1), 2), round(np.nanpercentile(self.result, 99), 2))
                else:
                    im.set_clim(1, rugositycolormax)
                ax.set_title(f'Arc-Chord Ratio (ACR) Rugosity Index\n(~{round(self.window_size_m, 2)}m x ~{round(self.window_size_m, 2)}m window)')
            else:
                im = ax.imshow(self.result, cmap='plasma')
                if rugositycolormax is None:
                    im.set_clim(round(np.nanpercentile(self.result, 1), 2), round(np.nanpercentile(self.result, 99), 2))
                else:
                    im.set_clim(1, rugositycolormax)
                ax.set_title(f'Conventional Rugosity Index\n(~{round(self.window_size_m, 2)}m x ~{round(self.window_size_m, 2)}m window)')
            ax.set_xlabel(f'X-axis grids \n(grid size ≈ {round(gridsize[0],4)} [{Zunit}])')
            ax.set_ylabel(f'Y-axis grids \n(grid size ≈ {-round(gridsize[4],4)} [{Zunit}])')
            cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.045, pad=0.13)

        plt.tight_layout()
        
        if savefig:
            if self.slope_correction:
                output_filename = os.path.splitext(input_file)[0] + f'_pyRugosity_ACR({round(self.window_size_m, 2)}m).png'
            else:
                output_filename = os.path.splitext(input_file)[0] + f'_pyRugosity_Conv({round(self.window_size_m, 2)}m).png'
            output_path = os.path.join(base_dir, output_filename)
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            print(f"Figure saved as '{os.path.basename(output_path)}'")

        if figshow:
            plt.show()
        else:
            plt.close(fig)