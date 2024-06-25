# pyTopoComlexity (v0.5.4)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11239338.svg)](https://doi.org/10.5281/zenodo.11239338)
![pyTopoComplexity cover](pyTopoComplexity.png)

This repository contains a set of codes for
1. measuring the topographic complexity (i.e., surface roughness) of a land surface, and
2. simulating topographic smoothing by non-linear hillslope diffusion processes.

## Citation

To use this code, please cite the Zenodo repository that hosts the latest release of this code: 
* Lai, L. S.-H. (2024). pyTopoComplexity. Zenodo. https://doi.org/10.5281/zenodo.11239338

## 2D-CWT Measurement of Topopraphic Complexity

There are three Jupyter Notebook files (see table below) using two-dimensional continuous wavelet transform (2D-CWT) with a Mexican Hat wevalet to measure the topographic complexity (i.e., surface roughness) of a land surface from a Digital Elevation Model (DEM). Such method quanitfy the wavelet-based curvature of the surface, which has been proposed to be a effective geomorphic metric for relative age dating of deep-seated landslide deposits, allowing a quick assessment of landslide freqency and spatiotemporal pattern over a large area.

| Code Files  | Descriptions |
| ------------- | ------------- |
| pyMexicanHat.ipynb  | The base version.  |
| pyMexicanHat_chunk.ipynb  | This version is developed to mitigate the RAM issues when handling large GeoTIFF files.  |
| pyMexicanHat_batch.ipynb  | This version is developed for batch-processing a large amount of raster files in the same directory. Chunk-processing optimization is included to mitigate the RAM issues when handling large GeoTIFF files.  |

The original MATLAB code was developed by Dr. Adam M. Booth (Portland State Univeristy) and used in Booth et al. (2009) and Booth et al. (2017). This MATLAB code was later revised and adapted by Dr. Sean R. LaHusen (Univeristy of Washington) and Erich N. Herzig (Univeristy of Washington) in their research (e.g., LaHusen et al., 2020; Herzig et al., 2023). Since November 2023, Dr. Larry Syu-Heng Lai (Univeristy of Washington) translated the code into a optimized open-source Python version.

The current codes have the capability to automoatically detect the grid spacing and the unit of XYZ directions (must be in feet or meters) of the input DEM raster, which can compute the 2D-CWT result with an proper wavelet scale factor at an designated Mexican Hat wavelet.

## Landform Smoothing via Nonlinear Hillslope Diffusion Modeling

The following Jupyter Notebook demonstrates the use of [Landlab](https://landlab.readthedocs.io/en/latest/index.html), a open-source Python framework for simulating landscape evolution, to model topographic smoothing driven by near-surface soil disturbance and downslope soil creep processes. Specifically, this notebook employs the `TaylorNonLinearDiffuser` component from LandLab, described as one element in the **terrainBento** package (Barnhart et al., 2019), to simulate topographic smoothing over time through non-linear hillslope diffusion processes (Roering et al., 1999).

| Code Files  | Descriptions |
| ------------- | ------------- |
| NonlinearDiff_LandLab.ipynb  | Using Lanblab component [TaylorNonLinearDiffuser](https://landlab.readthedocs.io/en/latest/reference/components/taylor_nonlinear_hillslope_flux.html) |

The current codes have the capability to automoatically detect the grid spacing and the unit of XYZ directions (must be in feet or meters) of the input DEM raster, which can convert the unit for diffusion coefficient (K) accordingly. 

**WARNING**: There is a known/unresolved stability issue when running `TaylorNonLinearDiffuser` component with a DEM with reprojected coordinate reference system (CRS) through GIS softwares. When using the example DEM, users may only use the original DEM with CRS: NAD83/Washington South (ftUS) (EPSG: 2286) and Z unit in US survey feet (e.g., the DEM files named with *"_f_3ftgrid"* or *"_f_6ftgrid"*)..

## Example DEM Raster Files

The example rasters include the LiDAR DEM files that cover the area and nearby region of a deep-seated landslide occurred in 2014 at Oso area of the North Fork Stillaguamish River (NFSR) valley, Washington State, USA. The example DEMs have various grid size, coordinate reference system (CRS), and unit of grid value (elevation, Z).

The example DEM files include:

| LiDAR DEM Files  | CRS | XY Grid Size | Z Unit | Descriptions |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Ososlid2014_f_3ftgrid.tif | NAD83/Washington South (ftUS) (EPSG: 2286) | 3.0 [US survey feet] | US survey feet | 2014 Oso Landslide |
| Ososlid2014_f_6ftgrid.tif | NAD83/Washington South (ftUS) (EPSG: 2286) | 6.0 [US survey feet] | US survey feet | 2014 Oso Landslide |
| Ososlid2014_m_3ftgrid.tif | NAD83/Washington South (EPSG: 32149) | ~0.9144 [meters] | meters | 2014 Oso Landslide |
| Ososlid2014_m_6ftgrid.tif | NAD83/Washington South (EPSG: 32149) | ~1.8288 [meters] | meters | 2014 Oso Landslide |
| Osoarea2014_f_6ftgrid.tif | NAD83/Washington South (ftUS) (EPSG: 2286) | 6.0 [US survey feet] | US survey feet | 2014 Oso Landslide & nearby NFSR valley |
| Osoarea2014_m_6ftgrid.tif | NAD83/Washington South (EPSG: 32149) | ~1.8288 [meters] | meters | 2014 Oso Landslide & nearby NFSR valley |

When testing the codes with the example DEM files, users should place the whole ***Example DEM*** subfolder in the same directory as the Jupyter Notebook files.

## References

*Journal Articles:*

* Barnhart, K., Glade, R., Shobe, C., Tucker, G. (2019). Terrainbento 1.0: a Python package for multi-model analysis in long-term drainage basin evolution. Geoscientific Model Development  12(4), 1267-1297. https://doi.org/10.5194/gmd-12-1267-2019
* Booth, A.M., LaHusen, S.R., Duvall, A.R., Montgomery, D.R., 2017. Holocene history of deep-seated landsliding in the North Fork Stillaguamish River valley from surface roughness analysis, radiocarbon dating, and numerical landscape evolution modeling. Journal of Geophysical Research: Earth Surface 122, 456-472. https://doi.org/10.1002/2016JF003934 
* Booth, A.M., Roering, J.J., Perron, J.T., 2009. Automated landslide mapping using spectral analysis and high-resolution topographic data: Puget Sound lowlands, Washington, and Portland Hills, Oregon. Geomorphology 109, 132-147. https://doi.org/10.1016/j.geomorph.2009.02.027   
* Ganti, V., Passalacqua, P., Foufoula-Georgiou, E. (2012). A sub-grid scale closure for nonlinear hillslope sediment transport models. Journal of Geophysical Research: Earth Surface, 117(F2). https://doi.org/10.1029/2011jf002181
* Herzig, E.N., Duvall, A.R., Booth, A.R., Stone, I., Wirth, E., LaHusen, S.R., Wartman, J., Grant, A., 2023. Evidence of Seattle Fault Earthquakes from Patterns in Deep‐Seated Landslides. Bulletin of the Seismological Society of America. https://doi.org/10.1785/0120230079
* Hobley, D.E.J., Adams, J.M., Nudurupati, S.S., Hutton, E.W.H., Gasparini, N.M., Istanbulluoglu, E., Tucker, G.E., 2017. Creative computing with Landlab: an open-source toolkit for building, coupling, and exploring two-dimensional numerical models of Earth-surface dynamics. Earth Surf. Dynam. 5, 21-46. https://doi.org/10.5194/esurf-5-21-2017
* LaHusen, S.R., Duvall, A.R., Booth, A.M., Grant, A., Mishkin, B.A., Montgomery, D.R., Struble, W., Roering, J.J., Wartman, J., 2020. Rainfall triggers more deep-seated landslides than Cascadia earthquakes in the Oregon Coast Range, USA. Science Advances 6, eaba6790. https://doi.org/10.1126/sciadv.aba6790
* Roering, J. J., Kirchner, J. W., & Dietrich, W. E. (1999). Evidence for nonlinear, diffusive sediment transport on hillslopes and implications for landscape morphology. Water Resources Research, 35(3), 853-870. https://doi.org/10.1029/1998wr900090

*Digital Elevation Model (DEM) Examples:*

* Washington Geological Survey, 2023. 'Stillaguamish 2014' and 'Snohoco Hazel 2006' projects [lidar data]: originally contracted by Washington State Department of Transportation (WSDOT). [accessed April 4, 2024, at http://lidarportal.dnr.wa.gov]


## Requirements
* os
* glob
* numpy
* scipy
* rasterio
* dask
* matplotlib
* ipywidgets
* landlab ([User Guide](https://landlab.readthedocs.io/en/latest/index.html))
  * Used components: TaylorNonLinearDiffuser, esri_ascii, imshowhs
* osgeo

## License
