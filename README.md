# pyTopoComlexity
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11239338.svg)](https://doi.org/10.5281/zenodo.11239338)
<p align="left">
 <img src="image/pyTopoComplexity_logo.png" width="30%" height="30%""/>
</p>

**pyTopoComplexity** is an open-source Python package designed to measure the topographic complexity (i.e., surface roughness) of land surfaces using digital elevation model (DEM) data. This package includes modules for three methods commonly used in the fields of geomorphology and oceanography for measuring topographic complexity, which are not fully available in Geographic Information System (GIS) software like QGIS.

| Modules  | Classes | Method Descriptions |
| ------------- | ------------- | ------------- |
| pycwtmexhat.py | CWTMexHat | Quanitfy the wavelet-based curvature of the terrain surface using two-dimensional continuous wavelet transform (2D-CWT) with a Mexican Hat wevalet |
| pyfracd.py | FracD | Conduct fractal dimension analysis on the terrain surface using variogram procedure |
| pyrugostiy.py | RugosityIndex | Calculate rugosity index of the terrain surface |

In this GitHub repository, each module has a corresponding example Jupyter Notebook file that includes detailed instructions on module usage and brief explanations of the applied theories with cited references. Example raster file data are included in the `~/example/` folder.

There is also an additional Jupyter Notebook, **nonlineardiff_Landlab.ipynb**, which leverages the power of [Landlab](https://landlab.readthedocs.io/en/latest/index.html) to perform forward simulation of landscape smoothing through non-linear hillslope diffusion process.

## Installation

```
pip install pytopocomplexity
```

## Citation

A manuscript is being prepared for submission to the [Journal of Open Source Software](https://joss.theoj.org). If you use pyTopoComplexity and the associated Jupyter Notebooks in your work, please consider citing the paper once it is accepted. In the meantime, users may cite the [Zenodo DOI](https://doi.org/10.5281/zenodo.11239338)).

## Modules for Surface Complexity Measurement

### 1. Two-Dimensional Continuous Wavelet Transform Analysis

```python
from pytopocomplexity import CWTMexHat
```

The module **pycwtmexhat.py** uses two-dimensional continuous wavelet transform (2D-CWT) with a Mexican Hat wevalet to measure the topographic complexity (i.e., surface roughness) of a land surface from a Digital Elevation Model (DEM). Such method quanitfy the wavelet-based curvature of the surface, which has been proposed to be a effective geomorphic metric for identifying and estimating the ages of historical deep-seated landslide deposits.

The method and early version of the code was developed by Dr. Adam M. Booth (Portland State Univeristy) in [2009](https://doi.org/10.1016/j.geomorph.2009.02.027), written in MATLAB (source code available from [Booth's personal website](https://web.pdx.edu/~boothad/tools.html)). This MATLAB code was later revised and adapted by Dr. Sean R. LaHusen (Univeristy of Washington) and Dr. Erich N. Herzig (Univeristy of Washington) in their research ([LaHusen et al., 2020](https://doi.org/10.1126/sciadv.aba6790); [Herzig et al. (2023)](https://doi.org/10.1785/0120230079)). Dr. Larry Syu-Heng Lai (Univeristy of Washington), under the supervision of Dr. Alison R. Duvall (Univeristy of Washington), translated the code into this optimized open-source Python version in 2024.

See **pycwtmexhat_example.ipynb** for detailed explanations and usage instructions.

<p align="center">
 <img src="image/CWTMexHat.png" width="100%" height="100%""/>
</p>

### 2. Fractal Dimentsion Analysis

```python
from pytopocomplexity import FracD
```

The **pyfracd.py** module calculates local fractal dimensions to assess topographic complexity. It also computes reliability parameters such as the standard error and the coefficient of determination (R²). The development of this module was greatly influenced by the Fortran code shared by Dr. Eulogio Pardo-Igúzquiza from his work in [Pardo-Igúzquiza and Dowd (2022)](https://doi.org/10.1016/j.icarus.2022.115109).

The local fractal dimension is determined by intersecting the surface within a moving window with four vertical planes in principal geographical directions, simplifying the problem to one-dimensional topographic profiles. The fractal dimension of these profiles is estimated using the variogram method, which models the relationship between dissimilarity and distance using a power-law function. While the fractal dimension value does not directly scale with the degree of surface roughness, smoother or more regular surfaces generally have lower fractal dimension values (closer to 2), whereas surfaces with higher fractal dimension values tend to be more complex or irregular. This method has been applied in terrain analysis for understanding spatial variability in surface roughness, classifying geomorphologic features, uncovering hidden spatial structures, and supporting geomorphological and geological mapping on Earth and other planetary bodies.

See **pyfracd_example.ipynb** for detailed explanations and usage instructions.

<p align="center">
 <img src="image/FracD.png" width="100%" height="100%""/>
</p>

### 3. Rugosity Index Calculation

```python
from pytopocomplexity import RugosityIndex
```

The module **pyrugosity.py** measure rugosity index of the land surface, which is widely used to assess landscape structural complexity.

By definition, the rugosity index has a minimum value of one, representing a completely flat surface. Typical values of the conventional rugosity index without slope correction ([Jenness, 2004](https://onlinelibrary.wiley.com/doi/abs/10.2193/0091-7648%282004%29032%5B0829%3ACLSAFD%5D2.0.CO%3B2)) range from one to three, although larger values are possible in very steep terrains. The slope-corrected rugosity index, also known as the Arc-Chord Ratio (ACR) rugosity index ([Du Preez, 2015](https://doi.org/10.1007/s10980-014-0118-8)), provides a better representation of local surface complexity. This method has been applied in classifying seafloor types by marine geologists and geomorphologists, studying small-scale hydrodynamics by oceanographers, and assessing available habitats in landscapes by ecologists and coral biologists.

See **pyrugosity_example.ipynb** for detailed explanations and usage instructions.

<p align="center">
 <img src="image/RugosityIndex.png" width="100%" height="100%""/>
</p>

## Forward Simulation of Landscape Smoothing through Nonlinear Hillslope Diffusion Process

In the `~/example/` folder, the Jupyter Notebook file **nonlineardiff_Landlab.ipynb** demonstrates the use of [Landlab](https://landlab.readthedocs.io/en/latest/index.html), an open-source Python framework for simulating landscape evolution, to model topographic smoothing driven by near-surface soil disturbance and downslope soil creep processes. Specifically, this notebook employs the [`TaylorNonLinearDiffuser`](https://landlab.readthedocs.io/en/latest/reference/components/taylor_nonlinear_hillslope_flux.html) component from [Landlab](https://landlab.readthedocs.io/en/latest/index.html), described as one element in the [`terrainBento`](https://github.com/TerrainBento/terrainbento) package, developed by [Barnhart et al. (2019)](https://gmd.copernicus.org/articles/12/1267/2019/), to simulate topographic smoothing over time through non-linear hillslope diffusion processes ([Roering et al., 1999](https://doi.org/10.1029/1998WR900090)).

Users need to define the diffusion coefficient (K) for the simulation. The code will automatically detect the units of the XYZ directions (must be in feet or meters) of the input DEM raster file and convert the unit for K accordingly.

<p align="center">
<img src="image/NonlinearDiff_demo.gif" width="65%" height="65%" align="center"/>
</p>

## Example DEM Raster Files

This repository include example LiDAR DEM files under `~/example/ExampleDEM/` that cover the area and nearby region of a deep-seated landslide occurred in 2014 at Oso area of the North Fork Stillaguamish River (NFSR) valley, Washington State, USA. The souce LiDAR DEM files were compiled from 'Stillaguamish 2014' and 'Snohoco Hazel 2006' projects that was originally contracted by Washington State Department of Transportation (WSDOT), downloaded from the [Washington Lidar Portal](http://lidarportal.dnr.wa.gov) on April 4, 2024.

A goal of this work allow users to reproduce the research by [Booth et al. (2017)](https://doi.org/10.1002/2016JF003934) and permit comparison of topographic complexity metrics derived from other regions using **pyTopoComplexity** package and the **nonlineardiff_Landlab.ipynb** simulation tools presented in this repository.

The example DEM raster files have various grid size, coordinate reference system (CRS), and unit of grid value (elevation, Z).

| LiDAR DEM Files  | CRS | XY Grid Size | Z Unit | Descriptions |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Ososlid2014_f_3ftgrid.tif | NAD83/Washington South (ftUS) (EPSG: 2286) | 3.0 [US survey feet] | US survey feet | 2014 Oso Landslide |
| Ososlid2014_m_3ftgrid.tif | NAD83/Washington South (EPSG: 32149) | ~0.9144 [meters] | meters | 2014 Oso Landslide |
| Ososlid2014_f_6ftgrid.tif | NAD83/Washington South (ftUS) (EPSG: 2286) | 6.0 [US survey feet] | US survey feet | 2014 Oso Landslide |
| Ososlid2014_m_6ftgrid.tif | NAD83/Washington South (EPSG: 32149) | ~1.8288 [meters] | meters | 2014 Oso Landslide |
| Osoarea2014_f_6ftgrid.tif | NAD83/Washington South (ftUS) (EPSG: 2286) | 6.0 [US survey feet] | US survey feet | 2014 Oso Landslide & nearby NFSR valley |

> [!NOTE]
> When testing the code with the example DEM files, users should place the entire `~/ExampleDEM/` subfolder in the same directory as the Jupyter Notebook files. Both the **pyTopoComplexity** package and the **nonlineardiff_landlab.ipynb** land-smoothing modeling tool have the capability to automatically detect the grid spacing and the units of the XYZ directions (must be in feet or meters) of the input DEM raster and compute the results in SI units.

## Requirements
For **pyTopoComplexity** package
* Python >= 3.10
* `numpy` >= 1.24
* `scipy` >= 1.10
* `rasterio` >= 1.3
* `dask` >= 2024.3
* `matplotlib` >= 3.7
* `tqdm` >= 4.66
* `numba` >= 0.57
* `statsmodels` >= 0.14

Additional packages for Jupyter Notebook examples:
* `pandas` >= 2.1
* `jupyter` >= 1.0

for landscape smoothing simulation
* [`landlab`](https://landlab.readthedocs.io/en/latest/index.html) >= 2.7
  * Used components: `TaylorNonLinearDiffuser`, `esri_ascii`, `imshowhs`
* `gdal`>= 3.6
* `ipywidgets` >= 8.1 [optional for interactive visualization]

See also the `environment.yml` file which can be used to create a virtual environment.

## License
**pyTopoComlexity** is licensed under the [Apache License 2.0](LICENSE).
