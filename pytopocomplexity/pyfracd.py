import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from numba import jit
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
from matplotlib.colors import LightSource

class FracD:
    """
    A class for calculating local fractal dimensions to assess topographic complexity.

    This class implements a methodology for fractal dimension analysis of Digital Elevation
    Model (DEM) data. It calculates local fractal dimensions and provides reliability
    parameters such as the standard error and coefficient of determination (R²).

    The local fractal dimension is determined by intersecting the surface within a moving
    window with four vertical planes in principal geographical directions. The fractal
    dimension of these profiles is estimated using the variogram method, which models the
    relationship between dissimilarity and distance using a power-law function.

    Required parameters:
    -----------
    window_size : int
        The size of the moving window for fractal dimension calculation.
    input_dir : str
        Path and filename of the input DEM file.
    fd2_output_dir : str
        Path and filename to save the fractal dimension output GeoTIFF file.
    se2_output_dir : str
        Path and filename to save the standard error output GeoTIFF file.
    r2_output_dir : str
        Path and filename to save the R² output GeoTIFF file.

    Attributes:
    -----------
    window_size : int
        Size of the moving window for fractal dimension calculation.
    chunk_processing : bool
        Whether to use chunk processing for large DEMs.
    chunksize : tuple
        Size of chunks for processing.
    fd2_out : numpy.ndarray
        Array to store calculated fractal dimensions.
    se2_out : numpy.ndarray
        Array to store standard errors of fractal dimensions.
    r2_out : numpy.ndarray
        Array to store R² values.
    ft2mUS : float
        Conversion factor from US survey feet to meters.
    ft2mInt : float
        Conversion factor from international feet to meters.
    window_size_m : float
        Window size converted in meters based on grid size.
    input_dir : str
        Path to the input DEM file.
    Z : numpy.ndarray
        Array storing the input DEM data.
    meta : dict
        Metadata of the input raster.

    Methods:
    --------
    analyze(input_dir)
        Perform fractal dimension analysis on the input DEM.
    export_results(fd2_output_dir, se2_output_dir, r2_output_dir)
        Export the analysis results to GeoTIFF files.
    plot_result()
        Plot and optionally save the analysis results.

    Example:
    --------
    >>> fa = FracD(window_size=10)
    >>> Z, fd_result, se_result, r2_result, window_m = fa.analyze('input_dem.tif')
    >>> fa.export_results('fd_output.tif', 'se_output.tif', 'r2_output.tif')
    >>> fa.plot_result()

    References:
    -----------
    Pardo-Igúzquiza, E., Dowd, P.A., 2022. The roughness of martian topography: A metre-scale
    fractal analysis of six selected areas. Icarus 384, 115109.
    https://doi.org/10.1016/j.icarus.2022.115109
    """

    def __init__(self, window_size=10):
        self.window_size = window_size
        self.chunk_processing = True
        self.chunksize = (512, 512)
        self.fd2_out = None
        self.se2_out = None
        self.r2_out = None
        self.ft2mUS = 1200/3937  # US survey foot to meter conversion factor
        self.ft2mInt = 0.3048    # International foot to meter conversion factor
        self.input_dir = None
        self.Z = None
        self.window_size_m = None
        self.meta = None

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def vario(Z, window_size, imo):
        """
        Calculate the variogram for a given profile.

        This method computes the variogram, which is a measure of spatial correlation
        in the data. It is used to estimate the fractal dimension of the profile.

        Parameters:
        -----------
        Z : numpy.ndarray
            1D array of elevation values.
        window_size : int
            Size of the moving window.
        imo : int
            Indicator for the profile direction (1 for orthogonal, 2 for diagonal).

        Returns:
        --------
        fd2 : float
            Estimated fractal dimension.
        se2 : float
            Standard error of the fractal dimension estimate.
        r2 : float
            Coefficient of determination (R²) of the variogram fit.
        """
        fac = 1.0 if imo == 1 else np.sqrt(2.0)
        x = np.arange(1, window_size + 1) * fac
        y = np.zeros(window_size)
        npa = np.zeros(window_size, dtype=np.int32)

        for i in range(window_size):
            diffs = Z[:-i-1] - Z[i+1:]
            y[i] = np.sum(diffs**2) / (2 * len(diffs))
            npa[i] = len(diffs)

        mask = npa > 0
        y = y[mask]
        x = x[mask]

        if len(x) < 2:
            return np.nan, np.nan, np.nan

        x = np.log(x)
        y = np.log(y)

        n = len(x)
        sumx = np.sum(x)
        sumy = np.sum(y)
        sumx2 = np.sum(x**2)
        sumxy = np.sum(x * y)

        b = (n * sumxy - sumx * sumy) / (n * sumx2 - sumx**2)
        a = (sumy - b * sumx) / n

        fd2 = 3.0 - b / 2.0
        yfit = a + b * x
        
        varb = (np.sum((y - yfit)**2) / (n - 2)) / (sumx2 - sumx**2 / n)
        se2 = np.sqrt(varb) / 2.0

        sum1 = np.sum((yfit - y)**2)
        sum2 = np.sum((y - np.mean(y))**2)
        r2 = 1.0 - sum1 / sum2

        return fd2, se2, r2
    
    def analyze(self, input_dir, variograms=True, chunk_processing=True, chunksize=(512, 512)):
        """
        Perform fractal dimension analysis on the input DEM.

        This method reads the input DEM, processes it to calculate fractal dimensions,
        standard errors, and R² values for each pixel, and optionally plots sample variograms.

        Parameters:
        -----------
        input_dir : str
            Path to the input DEM file.
        variograms : bool, optional
            Whether to plot sample variograms. Default is True.
        chunk_processing : bool, optional
            Whether to use chunk processing for large DEMs. Default is True.
        chunksize : tuple of int, optional
            Size of chunks for processing. Default is (512, 512).

        Returns:
        --------
        Z : numpy.ndarray
            The input elevation data.
        fd2_out : numpy.ndarray
            Calculated fractal dimensions.
        se2_out : numpy.ndarray
            Standard errors of fractal dimensions.
        r2_out : numpy.ndarray
            R² values.
        window_size_m : float
            Window size converted in meters based on grid size.
        """
        self.input_dir = input_dir
        self.chunk_processing = chunk_processing
        self.chunksize = chunksize
        with rasterio.open(input_dir) as src:
            self.meta = src.meta.copy()
            self.Z = src.read(1)
            Zunit = src.crs.linear_units
            self.fd2_out = np.full((src.height, src.width), np.nan, dtype=np.float32)
            self.se2_out = np.full((src.height, src.width), np.nan, dtype=np.float32)
            self.r2_out = np.full((src.height, src.width), np.nan, dtype=np.float32)

            # Get the cell size
            cell_size = src.transform[0]

            if any(unit in Zunit.lower() for unit in ["metre", "meter", "meters"]):
                pass
            elif any(unit in Zunit.lower() for unit in ["foot", "feet", "ft"]):
                if any(unit in Zunit.lower() for unit in ["us", "united states"]):
                    self.Z = self.Z * self.ft2mUS
                    cell_size = cell_size * self.ft2mUS
                else:
                    self.Z = self.Z * self.ft2mInt
                    cell_size = cell_size * self.ft2mInt
            else:
                raise ValueError("The unit of elevation 'Z' must be in feet or meters.")

            if not self.chunk_processing:
                self._process_non_chunked(self.Z, src.height, src.width)
            else:
                self._process_chunked(self.Z, src.height, src.width)

            # Replace edges with NaN
            fringeval = self.window_size // 2 + 1
            self.fd2_out[:fringeval, :] = np.nan
            self.fd2_out[:, :fringeval] = np.nan
            self.fd2_out[-fringeval:, :] = np.nan
            self.fd2_out[:, -fringeval:] = np.nan
            self.se2_out[:fringeval, :] = np.nan
            self.se2_out[:, :fringeval] = np.nan
            self.se2_out[-fringeval:, :] = np.nan
            self.se2_out[:, -fringeval:] = np.nan
            self.r2_out[:fringeval, :] = np.nan
            self.r2_out[:, :fringeval] = np.nan
            self.r2_out[-fringeval:, :] = np.nan
            self.r2_out[:, -fringeval:] = np.nan

            print(f"FD2 MIN : {np.nanmin(self.fd2_out)}")
            print(f"FD2 MAX : {np.nanmax(self.fd2_out)}")
            print(f"SE MIN : {np.nanmin(self.se2_out)}")
            print(f"SE MAX : {np.nanmax(self.se2_out)}")
            print(f"R2 MIN : {np.nanmin(self.r2_out)}")
            print(f"R2 MAX : {np.nanmax(self.r2_out)}")

        if variograms:
            self.plot_sample_variograms(self.Z)

        # Calculate window size in meters
        self.window_size_m = cell_size * self.window_size

        return self.Z, self.fd2_out, self.se2_out, self.r2_out, self.window_size_m

    def _process_non_chunked(self, Z, height, width):
        """
        Process the entire DEM without chunking.

        This method calculates fractal dimensions, standard errors, and R² values
        for each pixel in the DEM without dividing it into chunks.

        Parameters:
        -----------
        Z : numpy.ndarray
            The input elevation data.
        height : int
            Height of the DEM.
        width : int
            Width of the DEM.
        """
        for j in tqdm(range(height), desc="Analyzing rows"):
            for i in range(width):
                results = []
                # Horizontal slice
                data = Z[j, max(0, i-self.window_size):min(i+self.window_size+1, width)]
                if len(data) > self.window_size:
                    results.append(self.vario(data, self.window_size, 1))
                # Vertical slice
                data = Z[max(0, j-self.window_size):min(j+self.window_size+1, height), i]
                if len(data) > self.window_size:
                    results.append(self.vario(data, self.window_size, 1))
                # Diagonal slices
                for di, dj in [(1, 1), (1, -1)]:
                    data = []
                    for k in range(-self.window_size, self.window_size+1):
                        if 0 <= j+k*dj < height and 0 <= i+k*di < width:
                            data.append(Z[j+k*dj, i+k*di])
                    if len(data) > self.window_size:
                        results.append(self.vario(np.array(data), self.window_size, 2))
                if results:
                    fd2, se2, r2 = np.nanmean(results, axis=0)
                    self.fd2_out[j, i] = np.clip(fd2, 2.0, 3.0)
                    self.se2_out[j, i] = se2
                    self.r2_out[j, i] = r2

    def _process_chunked(self, Z, height, width):
        """
        Process the DEM in chunks for improved memory efficiency.

        This method divides the DEM into overlapping chunks and processes them in parallel,
        which can significantly reduce memory usage for large DEMs.

        Parameters:
        -----------
        Z : numpy.ndarray
            The input elevation data.
        height : int
            Height of the DEM.
        width : int
            Width of the DEM.
        """
        chunk_height, chunk_width = self.chunksize
        overlap = self.window_size // 2 + 1
        chunks = []
        for j in range(0, height, chunk_height - 2*overlap):
            for i in range(0, width, chunk_width - 2*overlap):
                chunk = Z[max(0, j-overlap):min(j+chunk_height+overlap, height),
                          max(0, i-overlap):min(i+chunk_width+overlap, width)]
                chunks.append((chunk, i, j))

        with tqdm(total=height, desc="Analyzing rows") as pbar:
            with Pool(cpu_count()) as pool:
                for fd2, se2, r2, i, j in pool.imap(self._process_chunk_parallel, chunks):
                    h, w = fd2.shape
                    non_overlap_h = max(0, h - 2*overlap)
                    non_overlap_w = max(0, w - 2*overlap)
                    self.fd2_out[j+overlap:j+overlap+non_overlap_h, i+overlap:i+overlap+non_overlap_w] = fd2[overlap:overlap+non_overlap_h, overlap:overlap+non_overlap_w]
                    self.se2_out[j+overlap:j+overlap+non_overlap_h, i+overlap:i+overlap+non_overlap_w] = se2[overlap:overlap+non_overlap_h, overlap:overlap+non_overlap_w]
                    self.r2_out[j+overlap:j+overlap+non_overlap_h, i+overlap:i+overlap+non_overlap_w] = r2[overlap:overlap+non_overlap_h, overlap:overlap+non_overlap_w]
                    pbar.update(non_overlap_h)

    def _process_chunk_parallel(self, args):
        """
        Process a single chunk of the DEM.

        This method is designed to be run in parallel, calculating fractal dimensions,
        standard errors, and R² values for each pixel in a given chunk of the DEM.

        Parameters:
        -----------
        args : tuple
            A tuple containing (Z_chunk, offset_i, offset_j), where:
            - Z_chunk is a numpy.ndarray of the DEM chunk
            - offset_i and offset_j are the offsets of the chunk in the full DEM

        Returns:
        --------
        fd2_out : numpy.ndarray
            Fractal dimensions for the chunk.
        se2_out : numpy.ndarray
            Standard errors for the chunk.
        r2_out : numpy.ndarray
            R² values for the chunk.
        offset_i : int
            X-offset of the chunk in the full DEM.
        offset_j : int
            Y-offset of the chunk in the full DEM.
        """
        Z, offset_i, offset_j = args
        fd2_out = np.full(Z.shape, np.nan, dtype=np.float32)
        se2_out = np.full(Z.shape, np.nan, dtype=np.float32)
        r2_out = np.full(Z.shape, np.nan, dtype=np.float32)

        for j in range(Z.shape[0]):
            for i in range(Z.shape[1]):
                results = []
                
                # Horizontal slice
                data = Z[j, max(0, i-self.window_size):min(i+self.window_size+1, Z.shape[1])]
                if len(data) > self.window_size:
                    results.append(self.vario(data, self.window_size, 1))

                # Vertical slice
                data = Z[max(0, j-self.window_size):min(j+self.window_size+1, Z.shape[0]), i]
                if len(data) > self.window_size:
                    results.append(self.vario(data, self.window_size, 1))

                # Diagonal slices
                for di, dj in [(1, 1), (1, -1)]:
                    data = []
                    for k in range(-self.window_size, self.window_size+1):
                        if 0 <= j+k*dj < Z.shape[0] and 0 <= i+k*di < Z.shape[1]:
                            data.append(Z[j+k*dj, i+k*di])
                    if len(data) > self.window_size:
                        results.append(self.vario(np.array(data), self.window_size, 2))

                if results:
                    fd2, se2, r2 = np.nanmean(results, axis=0)
                    fd2_out[j, i] = np.clip(fd2, 2.0, 3.0)
                    se2_out[j, i] = se2
                    r2_out[j, i] = r2

        return fd2_out, se2_out, r2_out, offset_i, offset_j
    
    def plot_sample_variograms(self, Z):
        """
        Plot sample variograms for the middle row, column, and diagonal of the DEM.

        This method provides a visual representation of the variograms used in the
        fractal dimension calculation, which can be useful for understanding the
        spatial structure of the DEM.

        Parameters:
        -----------
        Z : numpy.ndarray
            The input elevation data.
        """
        middle_row = Z.shape[0] // 2
        horizontal_slice = Z[middle_row, max(0, Z.shape[1]//2-self.window_size):min(Z.shape[1]//2+self.window_size+1, Z.shape[1])]
        self.plot_variogram(horizontal_slice, imo=1, title="Horizontal Variogram (Middle Row)")
        
        middle_col = Z.shape[1] // 2
        vertical_slice = Z[max(0, Z.shape[0]//2-self.window_size):min(Z.shape[0]//2+self.window_size+1, Z.shape[0]), middle_col]
        self.plot_variogram(vertical_slice, imo=1, title="Vertical Variogram (Middle Column)")
        
        diagonal_slice = [Z[i, i] for i in range(max(0, Z.shape[0]//2-self.window_size), min(Z.shape[0]//2+self.window_size+1, min(Z.shape)))]
        self.plot_variogram(diagonal_slice, imo=2, title="Diagonal Variogram")

    def plot_variogram(self, Z, imo, title=None):
        """
        Plot a variogram for a given slice of the DEM.

        This method calculates and plots the variogram for a given 1D slice of the DEM,
        which helps visualize the spatial correlation structure used in fractal dimension estimation.

        Parameters:
        -----------
        Z : numpy.ndarray
            1D array of elevation values.
        imo : int
            Indicator for the profile direction (1 for orthogonal, 2 for diagonal).
        title : str, optional
            Title for the plot. If None, a default title is used.
        """
        Z = np.array(Z)
        fac = 1.0 if imo == 1 else np.sqrt(2.0)
        x = np.arange(1, self.window_size + 1) * fac
        y = np.zeros(self.window_size)
        npa = np.zeros(self.window_size, dtype=np.int32)

        for i in range(self.window_size):
            diffs = Z[:-i-1] - Z[i+1:]
            y[i] = np.sum(diffs**2) / (2 * len(diffs))
            npa[i] = len(diffs)

        mask = npa > 0
        y = y[mask]
        x = x[mask]

        if len(x) < 2:
            print("Not enough data points for variogram plot")
            return

        plt.figure(figsize=(5, 3))
        plt.scatter(np.log(x), np.log(y), color='blue', label='Data')
        
        coef = np.polyfit(np.log(x), np.log(y), 1)
        poly1d_fn = np.poly1d(coef)
        plt.plot(np.log(x), poly1d_fn(np.log(x)), '--r', label='Fitted Line')

        plt.xlabel('Log(Lag Distance)')
        plt.ylabel('Log(Semivariance)')
        plt.title(title or 'Variogram Plot')
        plt.legend()
        plt.grid(True)
        plt.show()

        slope = coef[0]
        fd2 = 3.0 - slope / 2.0
        print(f"Estimated Fractal Dimension: {fd2}")

    def export_results(self, fd2_output_dir, se2_output_dir, r2_output_dir):
        """
        Export the analysis results to GeoTIFF files.

        This method saves the calculated fractal dimensions, standard errors, and R² values
        as separate GeoTIFF files, preserving the georeferencing information from the input DEM.

        Parameters:
        -----------
        fd2_output_dir : str
            Path to save the fractal dimension results.
        se2_output_dir : str
            Path to save the standard error results.
        r2_output_dir : str
            Path to save the R² results.

        Raises:
        -------
        ValueError
            If the metadata is missing and results cannot be exported.
        """
        if self.meta is None:
            raise ValueError("Metadata is missing. Cannot export the result.")
        
        self.meta.update(dtype=rasterio.float32, count=1, compress='deflate', bigtiff='IF_SAFER')

        with rasterio.open(fd2_output_dir, 'w', **self.meta) as dst:
            dst.write(self.fd2_out.astype(rasterio.float32), 1)
        print(f"'{os.path.basename(fd2_output_dir)}' is saved")

        with rasterio.open(se2_output_dir, 'w', **self.meta) as dst:
            dst.write(self.se2_out.astype(rasterio.float32), 1)
        print(f"'{os.path.basename(se2_output_dir)}' is saved")

        with rasterio.open(r2_output_dir, 'w', **self.meta) as dst:
            dst.write(self.r2_out.astype(rasterio.float32), 1)
        print(f"'{os.path.basename(r2_output_dir)}' is saved")

    def plot_result(self, output_dir=None, savefig=True, figshow=True, showhillshade=True, showse=True, showr2=True, fdcolormax=None, secolormax=None, r2colormax=None):
        """
        Plot the original DEM and the fractal dimension analysis results side by side.

        This method creates a plot showing:
        1. The original DEM as a hillshade (optional)
        2. The calculated fractal dimensions
        3. The standard errors of the fractal dimension estimates (optional)
        4. The R² values of the variogram fits (optional)

        Parameters:
        -----------
        output_dir : str, optional
            Specified directory to save the figure. If None, uses the input file's directory.
        savefig : bool, optional
            Whether to save the figure as a PNG file. Default is True.
        figshow : bool, optional
            Whether to display the figure. Default is True.
        showhillshade : bool, optional
            Whether to show the hillshade plot alongside the fractal dimension data. Default is True.
        showse : bool, optional
            Whether to show the standard error plot. Default is True.
        showr2 : bool, optional
            Whether to show the R² plot. Default is True.
        fdcolormax : float, optional
            Maximum value for fractal dimension color scale. If None, uses data-derived values.
        secolormax : float, optional
            Maximum value for standard error color scale. If None, uses data-derived values.
        r2colormax : float, optional
            Maximum value for R² color scale. If None, uses data-derived values.

        Raises:
        -------
        ValueError
            If the analysis hasn't been run before calling this method or if both savefig and figshow are False.
        """
        if self.Z is None or self.fd2_out is None or self.input_dir is None:
            raise ValueError("Analysis must be run before plotting results.")

        if not savefig and not figshow:
            raise ValueError("At least one of savefig or figshow must be True.")

        input_file = os.path.basename(self.input_dir)
        base_dir = output_dir if output_dir else os.path.dirname(self.input_dir)

        with rasterio.open(self.input_dir) as src:
            gridsize = src.transform
            Zunit = src.crs.linear_units

        num_plots = sum([showhillshade, True, showse, showr2])  # Always include fractal dimension plot
        fig, axes = plt.subplots(nrows=1, ncols=num_plots, figsize=(6*num_plots, 6))

        if num_plots == 1:
            axes = [axes]

        plot_index = 0

        if showhillshade:
            # Plot the hillshade
            ls = LightSource(azdeg=315, altdeg=45)
            hillshade = ls.hillshade(self.Z, vert_exag=2)
            hs = axes[plot_index].imshow(hillshade, cmap='gray')
            axes[plot_index].set_title(input_file)
            axes[plot_index].set_xlabel(f'X-axis grids \n(grid size ≈ {round(gridsize[0],4)} [{Zunit}])')
            axes[plot_index].set_ylabel(f'Y-axis grids \n(grid size ≈ {-round(gridsize[4],4)} [{Zunit}])')
            cbar = fig.colorbar(hs, ax=axes[plot_index], orientation='horizontal', fraction=0.045, pad=0.13)
            cbar.ax.set_visible(False)
            plot_index += 1

        # Plot the Fractal Dimension
        im1 = axes[plot_index].imshow(self.fd2_out, cmap='viridis')
        if fdcolormax is None:
            im1.set_clim(2, 3)
        else:
            im1.set_clim(2, fdcolormax)
        axes[plot_index].set_title(f'Fractal Dimension (~{round(self.window_size_m, 2)}m x ~{round(self.window_size_m, 2)}m window)')
        axes[plot_index].set_xlabel(f'X-axis grids \n(grid size ≈ {round(gridsize[0],4)} [{Zunit}])')
        axes[plot_index].set_ylabel(f'Y-axis grids \n(grid size ≈ {-round(gridsize[4],4)} [{Zunit}])')
        fig.colorbar(im1, ax=axes[plot_index], orientation='horizontal', fraction=0.045, pad=0.13)
        plot_index += 1

        if showse:
            # Plot the Standard Error
            im2 = axes[plot_index].imshow(self.se2_out, cmap='plasma')
            if secolormax is None:
                im2.set_clim(round(np.nanpercentile(self.se2_out, 0), 2), round(np.nanpercentile(self.se2_out, 100), 2))
            else:
                im2.set_clim(0, secolormax)
            axes[plot_index].set_title('Standard Error of Fractal Dimension')
            axes[plot_index].set_xlabel(f'X-axis grids \n(grid size ≈ {round(gridsize[0],4)} [{Zunit}])')
            axes[plot_index].set_ylabel(f'Y-axis grids \n(grid size ≈ {-round(gridsize[4],4)} [{Zunit}])')
            fig.colorbar(im2, ax=axes[plot_index], orientation='horizontal', fraction=0.045, pad=0.13)
            plot_index += 1

        if showr2:
            # Plot the r-square
            im3 = axes[plot_index].imshow(self.r2_out, cmap='coolwarm_r')
            if r2colormax is None:
                im3.set_clim(0, 1)
            else:
                im3.set_clim(0, r2colormax)
            axes[plot_index].set_title('Coefficient of determination (R$^{2}$)')
            axes[plot_index].set_xlabel(f'X-axis grids \n(grid size ≈ {round(gridsize[0],4)} [{Zunit}])')
            axes[plot_index].set_ylabel(f'Y-axis grids \n(grid size ≈ {-round(gridsize[4],4)} [{Zunit}])')
            fig.colorbar(im3, ax=axes[plot_index], orientation='horizontal', fraction=0.045, pad=0.13)

        plt.tight_layout()
        
        if savefig:
            output_filename = os.path.splitext(input_file)[0] + f'_pyFD({round(self.window_size_m, 2)}m).png'
            output_path = os.path.join(base_dir, output_filename)
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            print(f"Figure saved as '{os.path.basename(output_path)}'")
        
        if figshow:
            plt.show()
        else:
            plt.close(fig)