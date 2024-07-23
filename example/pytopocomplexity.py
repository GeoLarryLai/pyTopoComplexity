import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import dask.array as da
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm
from numba import jit
from scipy.ndimage import generic_filter
from scipy.signal import fftconvolve, convolve2d

class pycwtmexhat:
    def __init__(self, conv_method='fft', chunk_processing=True, chunksize=(512,512)):
        self.chunk_processing = chunk_processing
        self.conv_method = conv_method
        self.chunksize = chunksize
        self.ft2mUS = 1200/3937  # US survey foot to meter conversion factor
        self.ft2mInt = 0.3048    # International foot to meter conversion factor

    def conv2_mexh(self, Z, s, Delta):
        X, Y = np.meshgrid(np.arange(-8 * s, 8 * s + 1), np.arange(-8 * s, 8 * s + 1))
        psi = (-1/(np.pi*(s * Delta)**4)) * (1 - (X**2 + Y**2)/(2 * s**2)) * np.exp(-(X**2 + Y**2)/(2* s**2))
        
        if self.conv_method == 'fft':
            C = (Delta**2) * fftconvolve(Z, psi, mode='same')
        elif self.conv_method == 'conv':
            C = (Delta**2) * convolve2d(Z, psi, mode='same')
        else:
            raise ValueError("Convolution method must be 'fft' or 'conv'.")
        
        return C

    def Delta_s_Calculate(self, input_dir, Lambda):
        with rasterio.open(input_dir) as src:
            transform = src.transform
            crs = src.crs

        if any(unit in crs.linear_units.lower() for unit in ["metre", "meter"]):
            Delta = np.mean([transform[0], -transform[4]])
        elif any(unit in crs.linear_units.lower() for unit in ["foot", "feet", "ft"]):
            if any(unit in crs.linear_units.lower() for unit in ["us", "united states"]):
                Delta = np.mean([transform[0] * self.ft2mUS, -transform[4] * self.ft2mUS])
            else:
                Delta = np.mean([transform[0] * self.ft2mInt, -transform[4] * self.ft2mInt])
        else:
            raise ValueError("The units of XY directions must be in feet or meters.")

        s = (Lambda/Delta)*((5/2)**(1/2)/(2*np.pi))

        return Delta, s

    def process_with_dask(self, input_dir, s, Delta):
        with rasterio.open(input_dir) as src:
            meta = src.meta.copy()
            Zunit = src.crs.linear_units
            Z = src.read(1)

        # Convert Z to meters if necessary
        if any(unit in Zunit.lower() for unit in ["metre", "meter", "meters"]):
            pass
        elif any(unit in Zunit.lower() for unit in ["foot", "feet", "ft"]):
            if any(unit in Zunit.lower() for unit in ["us", "united states"]):
                Z = Z * self.ft2mUS
            else:
                Z = Z * self.ft2mInt
        else:
            raise ValueError("The unit of elevation 'z' must be in feet or meters.")

        # Create Dask array from the converted Z
        dask_Z = da.from_array(Z, chunks=self.chunksize)

        processed_data = dask_Z.map_overlap(
            lambda block: np.abs(self.conv2_mexh(block, s, Delta)),
            depth=int(s * 4),
            boundary='reflect',
            trim=True,
            dtype=np.float32
        )

        # Compute the result with ProgressBar
        with ProgressBar():
            result = processed_data.compute()
        
        fringeval = int(np.ceil(s * 4))
        result[:fringeval, :] = np.nan
        result[:, :fringeval] = np.nan
        result[-fringeval:, :] = np.nan
        result[:, -fringeval:] = np.nan

        return Z, result, meta

    def process_mexhat(self, input_dir, s, Delta):
        with rasterio.open(input_dir) as src:
            Z = src.read(1)
            Zunit = src.crs.linear_units
            meta = src.meta.copy()

        if any(unit in Zunit.lower() for unit in ["metre", "meter", "meters"]):
            pass
        elif any(unit in Zunit.lower() for unit in ["foot", "feet", "ft"]):
            if any(unit in Zunit.lower() for unit in ["us", "united states"]):
                Z = Z * self.ft2mUS
            else:
                Z = Z * self.ft2mInt
        else:
            raise ValueError("The unit of elevation 'z' must be in feet or meters.")
        
        # Add progress bar
        total_iterations = Z.shape[0] * Z.shape[1]
        with tqdm(total=total_iterations, desc="Processing", unit="pixel") as pbar:
            C2 = self.conv2_mexh(Z, s, Delta)
            result = np.abs(C2)
            pbar.update(total_iterations)

        cropedge = np.ceil(s * 4)
        fringeval = int(cropedge)
        result[:fringeval, :] = np.nan
        result[:, :fringeval] = np.nan
        result[-fringeval:, :] = np.nan
        result[:, -fringeval:] = np.nan

        return Z, result, meta

    def analyze(self, input_dir, Lambda):
        if Lambda is None:
            raise ValueError("Lambda is missing. Cannot conduct the analysis.")
        
        #Derive the correct grid spacing and wavelet scale
        Delta, s = self.Delta_s_Calculate(input_dir, Lambda)
        
        if self.chunk_processing:
            Z, result, meta = self.process_with_dask(input_dir, s, Delta)
        else:
            Z, result, meta = self.process_mexhat(input_dir, s, Delta)
        
        return Z, result, meta

    def export_result(self, result, meta, output_dir):
        if meta is None:
            raise ValueError("Metadata is missing. Cannot export the result.")
        
        # Setup output metadata
        meta.update(dtype=rasterio.float32, count=1, compress='deflate', bigtiff='IF_SAFER')
        
        # Write result to GeoTIFF
        with rasterio.open(output_dir, 'w', **meta) as dst:
            dst.write(result.astype(rasterio.float32), 1)
        
        print(f"Processed result saved to {os.path.basename(output_dir)}")






class pyfracd:
    def __init__(self, npas=10):
        self.npas = npas
        self.fd2_out = None
        self.se2_out = None
        self.r2_out = None
        self.ft2mUS = 1200/3937  # US survey foot to meter conversion factor
        self.ft2mInt = 0.3048    # International foot to meter conversion factor

    @staticmethod
    @jit(nopython=True)
    def vario(z, npas, imo):
        fac = 1.0 if imo == 1 else np.sqrt(2.0)
        x = np.arange(1, npas + 1) * fac
        y = np.zeros(npas)
        npa = np.zeros(npas, dtype=np.int32)

        for i in range(npas):
            diffs = z[:-i-1] - z[i+1:]
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
    
    def analyze(self, input_dir, variograms=True):
        with rasterio.open(input_dir) as src:
            meta = src.meta.copy()
            Z = src.read(1)
            Zunit = src.crs.linear_units
            self.fd2_out = np.zeros((src.height, src.width), dtype=np.float32)
            self.se2_out = np.zeros((src.height, src.width), dtype=np.float32)
            self.r2_out = np.zeros((src.height, src.width), dtype=np.float32)

            # Convert Z to meters if necessary
            if any(unit in Zunit.lower() for unit in ["metre", "meter", "meters"]):
                pass
            elif any(unit in Zunit.lower() for unit in ["foot", "feet", "ft"]):
                if any(unit in Zunit.lower() for unit in ["us", "united states"]):
                    Z = Z * self.ft2mUS
                else:
                    Z = Z * self.ft2mInt
            else:
                raise ValueError("The unit of elevation 'z' must be in feet or meters.")

            for j in tqdm(range(src.height), desc="Analyzing rows"):
                for i in range(src.width):
                    results = []
                    
                    # Horizontal analysis
                    data = Z[j, max(0, i-20):min(i+21, src.width)]
                    if len(data) > self.npas:
                        results.append(self.vario(data, self.npas, 1))

                    # Vertical analysis
                    data = Z[max(0, j-20):min(j+21, src.height), i]
                    if len(data) > self.npas:
                        results.append(self.vario(data, self.npas, 1))

                    # Diagonal analyses
                    for di, dj in [(1, 1), (1, -1)]:
                        data = []
                        for k in range(-14, 15):
                            if 0 <= j+k*dj < src.height and 0 <= i+k*di < src.width:
                                data.append(Z[j+k*dj, i+k*di])
                        if len(data) > self.npas:
                            results.append(self.vario(np.array(data), self.npas, 2))

                    if results:
                        fd2, se2, r2 = np.nanmean(results, axis=0)
                        self.fd2_out[j, i] = np.clip(fd2, 2.0, 3.0)
                        self.se2_out[j, i] = se2
                        self.r2_out[j, i] = r2
            
            print(f"FD2 MIN : {np.nanmin(self.fd2_out)}")
            print(f"FD2 MAX : {np.nanmax(self.fd2_out)}")
            print(f"SE MIN : {np.nanmin(self.se2_out)}")
            print(f"SE MAX : {np.nanmax(self.se2_out)}")
            print(f"R2 MIN : {np.nanmin(self.r2_out)}")
            print(f"R2 MAX : {np.nanmax(self.r2_out)}")

        if variograms:
            self.plot_sample_variograms(Z)

        return Z, self.fd2_out, self.se2_out, self.r2_out, meta

    def plot_variogram(self, z, imo, title=None):
        """
        Plot the variogram for given data.
        
        :param z: 1D array or list of elevation data
        :param imo: Mode (1 for horizontal/vertical, 2 for diagonal)
        :param title: Optional title for the plot
        """
        z = np.array(z)  # Convert to numpy array if it's a list
        fac = 1.0 if imo == 1 else np.sqrt(2.0)
        x = np.arange(1, self.npas + 1) * fac
        y = np.zeros(self.npas)
        npa = np.zeros(self.npas, dtype=np.int32)

        for i in range(self.npas):
            diffs = z[:-i-1] - z[i+1:]
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
        
        # Fit line
        coef = np.polyfit(np.log(x), np.log(y), 1)
        poly1d_fn = np.poly1d(coef)
        plt.plot(np.log(x), poly1d_fn(np.log(x)), '--r', label='Fitted Line')

        plt.xlabel('Log(Lag Distance)')
        plt.ylabel('Log(Semivariance)')
        plt.title(title or 'Variogram Plot')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Calculate and print fractal dimension
        slope = coef[0]
        fd2 = 3.0 - slope / 2.0
        print(f"Estimated Fractal Dimension: {fd2}")

    def plot_sample_variograms(self, Z):
        # Plot variogram for a horizontal slice in the middle of the DEM
        middle_row = Z.shape[0] // 2
        horizontal_slice = Z[middle_row, max(0, Z.shape[1]//2-20):min(Z.shape[1]//2+21, Z.shape[1])]
        self.plot_variogram(horizontal_slice, imo=1, title="Horizontal Variogram (Middle Row)")
        
        # Plot variogram for a vertical slice
        middle_col = Z.shape[1] // 2
        vertical_slice = Z[max(0, Z.shape[0]//2-20):min(Z.shape[0]//2+21, Z.shape[0]), middle_col]
        self.plot_variogram(vertical_slice, imo=1, title="Vertical Variogram (Middle Column)")
        
        # Plot variogram for a diagonal slice
        diagonal_slice = [Z[i, i] for i in range(max(0, Z.shape[0]//2-14), min(Z.shape[0]//2+15, min(Z.shape)))]
        self.plot_variogram(diagonal_slice, imo=2, title="Diagonal Variogram")

    def export_results(self, fd2_output_dir, se2_output_dir, r2_output_dir, meta):
        if meta is None:
            raise ValueError("Metadata is missing. Cannot export the result.")
        
        # Setup output metadata
        meta.update(dtype=rasterio.float32, count=1, compress='deflate', bigtiff='IF_SAFER')

        # Write result to GeoTIFF
        with rasterio.open(fd2_output_dir, 'w', **meta) as dst:
            dst.write(self.fd2_out.astype(rasterio.float32), 1)
        print(f"Processed result saved to {os.path.basename(fd2_output_dir)}")

        with rasterio.open(se2_output_dir, 'w', **meta) as dst:
            dst.write(self.se2_out.astype(rasterio.float32), 1)
        print(f"Processed result saved to {os.path.basename(se2_output_dir)}")

        with rasterio.open(r2_output_dir, 'w', **meta) as dst:
            dst.write(self.r2_out.astype(rasterio.float32), 1)
        print(f"Processed result saved to {os.path.basename(r2_output_dir)}")






class pyrugosity:
    def __init__(self, chunk_processing=False, chunksize=(512, 512)):
        self.ft2mUS = 1200/3937  # US survey foot to meter conversion factor
        self.ft2mInt = 0.3048    # International foot to meter conversion factor
        self.chunk_processing = chunk_processing
        self.chunksize = chunksize

    @staticmethod
    @jit(nopython=True)
    def compute_triangle_areas(values, cell_size):
        window_size = int(np.sqrt(len(values)))
        center = values[len(values) // 2]
        total_surface_area = 0.0
        
        for i in range(window_size - 1):
            for j in range(window_size - 1):
                idx = i * window_size + j
                corner_dist = np.sqrt(2 * cell_size**2)
                
                # Calculate edges
                orthogonal_edges = np.sqrt((center - values[idx])**2 + cell_size**2) / 2
                diagonal_edges = np.sqrt((center - values[idx])**2 + corner_dist**2) / 2
                adjacent_edges = np.sqrt((values[idx] - values[idx + 1])**2 + cell_size**2) / 2
                
                # Calculate triangle areas using Heron's formula
                s = (orthogonal_edges + diagonal_edges + adjacent_edges) / 2
                area = np.sqrt(s * (s - orthogonal_edges) * (s - diagonal_edges) * (s - adjacent_edges))
                # Handle potential numerical instability
                if area < 0:
                    return 0
                total_surface_area += area * 8  # Eight triangles per sub-square
        
        return total_surface_area

    def calculate_rugosity(self, Z, cell_size, window_size):
        if window_size % 2 == 0 or window_size < 3:
            raise ValueError("'window_size' must be an odd integer >= 3.")

        def rugosity_filter(values):
            surface_area = self.compute_triangle_areas(values, cell_size)
            planar_area = ((window_size - 1) * cell_size)**2
            return surface_area / planar_area

        if self.chunk_processing:
            # Use dask for parallel processing with user-defined chunk size
            dask_Z = da.from_array(Z, chunks=self.chunksize)
            rugosity = dask_Z.map_overlap(
                lambda block: generic_filter(block, rugosity_filter, size=window_size, mode='nearest'),
                depth=window_size//2,
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
                result = generic_filter(Z, progress_filter, size=window_size, mode='nearest')
        
        return result

    def analyze(self, input_path, window_size=3):
        with rasterio.open(input_path) as src:
            meta = src.meta.copy()
            
            # Read the raster data
            Z = src.read(1)
            Zunit = src.crs.linear_units
            
            # Get the cell size
            cell_size = src.transform[0]
            
            # Convert to meters if necessary            
            if any(unit in Zunit.lower() for unit in ["metre", "meter", "meters"]):
                pass
            elif any(unit in Zunit.lower() for unit in ["foot", "feet", "ft"]):
                if any(unit in Zunit.lower() for unit in ["us", "united states"]):
                    Z = Z * self.ft2mUS
                    cell_size = cell_size * self.ft2mUS
                else:
                    Z = Z * self.ft2mInt
                    cell_size = cell_size * self.ft2mInt
            else:
                raise ValueError("The unit of elevation 'z' must be in feet or meters.")

            # Calculate rugosity
            rugosity = self.calculate_rugosity(Z, cell_size, window_size)        

            # Calculate window size in meters
            window_size_m = cell_size * window_size

        return Z, rugosity, meta, window_size_m

    def export_result(self, result, meta, output_dir):
        if meta is None:
            raise ValueError("Metadata is missing. Cannot export the result.")
        
        # Setup output metadata
        meta.update(dtype=rasterio.float32, count=1, compress='deflate', bigtiff='IF_SAFER')
        
        # Write result to GeoTiff
        with rasterio.open(output_dir, 'w', **meta) as dst:
            dst.write(result.astype(rasterio.float32), 1)
        
        print(f"Processed result saved to {os.path.basename(output_dir)}")
