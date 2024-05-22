# pyTopoComlexity (v0.5)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10065283.svg)](https://doi.org/10.5281/zenodo.10065283)

This repository contains a set of codes for measuring the topographic complexity (i.e., surface roughness) of a land surface using two-dimensional continuous wavelet transform (2D-CWT) with a Mexican Hat wevalet. Such method quanitfy the wavelet-based curvature of the surface, which has been proposed to be a effective geomorphic metric for relative age dating of deep-seated landslide deposits, allowing a quick assessment of landslide freqency and spatiotemporal pattern over a large scale area.

There are three Jupyter Notebook files:

| Code Files  | Descriptions |
| ------------- | ------------- |
| pyMexicanHat.ipynb  | The base version.  |
| pyMexicanHat_chunk.ipynb  | This version is developed to mitigate the RAM issues when handling large GeoTIFF files.  |
| pyMexicanHat_batch.ipynb  | This version is developed for batch-processing a large amount of raster files in the same directory. Chunk-processing optimization is included to mitigate the RAM issues when handling large GeoTIFF files.  |

The example rasters include the LiDAR Digital Elevation Model (DEM) files that cover the area and nearby region of a deep-seated landslide occurred in 2014 at Oso area of the North Fork Stillaguamish River (NFSR) valley, Washington State, USA. The example DEMs have various grid size, coordinate reference system (CRS), and unit of grid value (elevation, Z).

The example DEM files include:

| LiDAR DEM Files  | CRS | XY Grid Size | Z Unit | Descriptions |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Ososlid2014_f_3ftgrid.tif | NAD83/Washington South (ftUS) (EPSG: 2286) | 3.0 [US survey feet] | US survey feet | 2014 Oso Landslide |
| Ososlid2014_f_6ftgrid.tif | NAD83/Washington South (ftUS) (EPSG: 2286) | 6.0 [US survey feet] | US survey feet | 2014 Oso Landslide |
| Ososlid2014_m_3ftgrid.tif | NAD83/Washington South (EPSG: 32149) | ~0.9144 [meters] | meters | 2014 Oso Landslide |
| Ososlid2014_m_6ftgrid.tif | NAD83/Washington South (EPSG: 32149) | ~1.8288 [meters] | meters | 2014 Oso Landslide |
| Osoarea2014_f_6ftgrid.tif | NAD83/Washington South (ftUS) (EPSG: 2286) | 6.0 [US survey feet] | US survey feet | 2014 Oso Landslide & nearby NFSR valley |
| Osoarea2014_m_6ftgrid.tif | NAD83/Washington South (EPSG: 32149) | ~1.8288 [meters] | meters | 2014 Oso Landslide & nearby NFSR valley |

The current codes have the capability to automoatically detect the grid spacing and the unit of XYZ directions (must be in feet or meters) of the input DEM raster, which can compute the 2D-CWT result with an proper wavelet scale factor at an designated Mexican Hat wavelet. When testing the codes with the example DEM files, users should decompress the Example DEM.zip and place the whole unarchived subfolder in the same directory as the Jupyter Notebook files.
