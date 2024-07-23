import os
import numpy as np
import rasterio
import dask.array as da
from dask.diagnostics import ProgressBar
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

        if any(unit in Zunit.lower() for unit in ["metre", "meter", "meters"]):
            pass
        elif any(unit in Zunit.lower() for unit in ["foot", "feet", "ft"]):
            if any(unit in Zunit.lower() for unit in ["us", "united states"]):
                Z = Z * self.ft2mUS
            else:
                Z = Z * self.ft2mInt
        else:
            raise ValueError("The unit of elevation 'z' must be in feet or meters.")

        dask_Z = da.from_array(Z, chunks=self.chunksize)

        processed_data = dask_Z.map_overlap(
            lambda block: np.abs(self.conv2_mexh(block, s, Delta)),
            depth=int(s * 4),
            boundary='reflect',
            trim=True,
            dtype=np.float32
        )

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
        
        C2 = self.conv2_mexh(Z, s, Delta)
        result = np.abs(C2)

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
        
        Delta, s = self.Delta_s_Calculate(input_dir, Lambda)
        
        if self.chunk_processing:
            Z, result, meta = self.process_with_dask(input_dir, s, Delta)
        else:
            Z, result, meta = self.process_mexhat(input_dir, s, Delta)
        
        return Z, result, meta

    def export_result(self, result, meta, output_dir):
        if meta is None:
            raise ValueError("Metadata is missing. Cannot export the result.")
        
        meta.update(dtype=rasterio.float32, count=1, compress='deflate', bigtiff='IF_SAFER')
        
        with rasterio.open(output_dir, 'w', **meta) as dst:
            dst.write(result.astype(rasterio.float32), 1)
        
        print(f"Processed result saved to {os.path.basename(output_dir)}")
