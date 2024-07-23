import os
import numpy as np
import rasterio
import dask.array as da
from dask.diagnostics import ProgressBar
from numba import jit
from scipy.ndimage import generic_filter
from tqdm.auto import tqdm

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
                total_surface_area += area * 4  # Four triangles per sub-square
        
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
