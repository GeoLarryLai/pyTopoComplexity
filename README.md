# pyTopoComplexity
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11239338.svg)](https://doi.org/10.5281/zenodo.11239338)  [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/pytopocomplexity.svg)](https://anaconda.org/conda-forge/pytopocomplexity)  [![Conda Version](https://img.shields.io/conda/vn/conda-forge/pytopocomplexity.svg)](https://anaconda.org/conda-forge/pytopocomplexity)
<p align="left">
 <img src="image/pyTopoComplexity_logo.png" width="30%" height="30%""/>
</p>

**pyTopoComplexity** is an open-source Python package designed to measure the topographic complexity (i.e., surface roughness) of land surfaces using digital elevation model (DEM) data. This package includes modules for **four** modern methods used to measure topographic complexity in the fields of geology, geomorphology, geography, ecology, and oceanography.

| Modules  | Classes | Method Descriptions |
| ------------- | ------------- | ------------- |
| pycwtmexhat.py | CWTMexHat | Quanitfies the wavelet-based curvature of the terrain surface using two-dimensional continuous wavelet transform (2D-CWT) with a Mexican Hat wevalet |
| pyfracd.py | FracD | Conducts fractal dimension analysis on the terrain surface using variogram procedure |
| pyrugostiy.py | RugosityIndex | Calculates Rugosity Index of the terrain surface |
| pytpi.py | TPI | Calculates Terrain Position Index of the topography |

In this repository, each module has a corresponding example Jupyter Notebook file that includes detailed instructions on module usage and brief explanations of the applied theories with cited references. Example DEM raster file data are included in the `~/ExampleDEM/` folder.

There is also an additional Jupyter Notebook, [**Landlab_simulation.ipynb**](https://github.com/GeoLarryLai/pyTopoComplexity/blob/main/Landlab_simulation.ipynb), which leverages the power of [Landlab](https://landlab.readthedocs.io/en/latest/index.html) to perform forward simulation of landscape smoothing through non-linear hillslope diffusion process.

## Citation

If you use **pyTopoComplexity** and the associated Jupyter Notebooks in your work, please cite the following paper:

Lai, L. S.-H., Booth, A. M., Duvall, A. R., and Herzig, E. (2025) Short Communication: Multiscale topographic complexity analysis with pyTopoComplexity. Earth Surface Dynamics, 13(3), 417-435. https://doi.org/10.5194/esurf-13-417-2025

If you have any questions, feedback, or interest in collaboration, feel free to reach out to me at larrysyuhenglai@gmail.com or larry.lai@beg.utexas.edu

## Installation

Users can install **pyTopoComplexity** directly from [PyPI](https://pypi.org/project/pytopocomplexity/) with `pip` command:
```
pip install pytopocomplexity
```
or from `conda-forge` repository with `conda`:
```
conda install pytopocomplexity
```
To run the simulations for landscape evolution in the [**Landlab_simulation.ipynb**](https://github.com/GeoLarryLai/pyTopoComplexity/blob/main/Landlab_simulation.ipynb), users need to install [Landlab](https://landlab.readthedocs.io/en/latest/index.html) in addition to **pyTopoComplexity**. Please visit [Landlab](https://landlab.readthedocs.io/en/latest/index.html)'s installation instruction on CSDMS website: https://landlab.readthedocs.io/en/latest/installation.html.

## Workflow

<p align="center">
 <img src="image/workflow-schematic.png" width="100%" height="100%""/>
</p>

## Modules for Surface Complexity Measurement

### 1. Two-Dimensional Continuous Wavelet Transform Analysis

```python
from pytopocomplexity import CWTMexHat
```

The module **pycwtmexhat.py** uses two-dimensional continuous wavelet transform (2D-CWT) with a Mexican Hat wevalet to measure the topographic complexity (i.e., surface roughness) of a land surface from a Digital Elevation Model (DEM). Such method quanitfy the wavelet-based curvature of the surface, which has been proposed to be a effective geomorphic metric for identifying and estimating the ages of historical deep-seated landslide deposits. The method and early version of the code was developed by Dr. Adam M. Booth (Portland State Univeristy) in [2009](https://doi.org/10.1016/j.geomorph.2009.02.027), written in MATLAB (source code available from [Booth's personal website](https://web.pdx.edu/~boothad/tools.html)). This MATLAB code was later revised and adapted by Dr. Sean R. LaHusen (Univeristy of Washington) and Dr. Erich N. Herzig (Univeristy of Washington) in their research ([LaHusen et al., 2020](https://doi.org/10.1126/sciadv.aba6790); [Herzig et al. (2023)](https://doi.org/10.1785/0120230079)). Dr. Larry Syu-Heng Lai (Univeristy of Washington), under the supervision of Dr. Alison R. Duvall (Univeristy of Washington), translated the code into this optimized open-source Python version in 2024.

See [**Example_pycwtmexhat.ipynb**](https://github.com/GeoLarryLai/pyTopoComplexity/blob/main/Example_pycwtmexhat.ipynb) for detailed explanations and usage instructions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GeoLarryLai/pyTopoComplexity/blob/main/Example_pycwtmexhat.ipynb)

<p align="center">
 <img src="image/cwtmexhat.png" width="100%" height="100%""/>
</p>

### 2. Fractal Dimentsion Analysis

```python
from pytopocomplexity import FracD
```

The **pyfracd.py** module calculates local fractal dimensions to assess topographic complexity. It also computes reliability parameters such as the standard error and the coefficient of determination (R²). The development of this module was greatly influenced by the Fortran code shared by Dr. Eulogio Pardo-Igúzquiza from his work in [Pardo-Igúzquiza and Dowd (2022)](https://doi.org/10.1016/j.icarus.2022.115109). The local fractal dimension is determined by intersecting the surface within a moving window with four vertical planes in principal geographical directions, simplifying the problem to one-dimensional topographic profiles. The fractal dimension of these profiles is estimated using the variogram method, which models the relationship between dissimilarity and distance using a power-law function. While the fractal dimension value does not directly scale with the degree of surface roughness, smoother or more regular surfaces generally have lower fractal dimension values (closer to 2), whereas surfaces with higher fractal dimension values tend to be more complex or irregular. This method has been applied in terrain analysis for understanding spatial variability in surface roughness, classifying geomorphologic features, uncovering hidden spatial structures, and supporting geomorphological and geological mapping on Earth and other planetary bodies.

See [**Example_pyfracd.ipynb**](https://github.com/GeoLarryLai/pyTopoComplexity/blob/main/Example_pyfracd.ipynb) for detailed explanations and usage instructions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GeoLarryLai/pyTopoComplexity/blob/main/Example_pyfracd.ipynb)


<p align="center">
 <img src="image/fracd.png" width="100%" height="100%""/>
</p>

### 3. Rugosity Index Calculation

```python
from pytopocomplexity import RugosityIndex
```

The module **pyrugosity.py** measure Rugosity Index of the land surface, which is widely used to assess landscape structural complexity. By definition, the Rugosity Index has a minimum value of one, representing a completely flat surface. Typical values of the conventional Rugosity Index without slope correction ([Jenness, 2004](https://onlinelibrary.wiley.com/doi/abs/10.2193/0091-7648%282004%29032%5B0829%3ACLSAFD%5D2.0.CO%3B2)) range from one to three, although larger values are possible in very steep terrains. The slope-corrected Rugosity Index, also known as the Arc-Chord Ratio (ACR) Rugosity Index ([Du Preez, 2015](https://doi.org/10.1007/s10980-014-0118-8)), provides a better representation of local surface complexity. This method has been applied in classifying seafloor types by marine geologists and geomorphologists, studying small-scale hydrodynamics by oceanographers, and assessing available habitats in landscapes by ecologists and coral biologists.

See [**Example_pyrugosity.ipynb**](https://github.com/GeoLarryLai/pyTopoComplexity/blob/main/Example_pyrugosity.ipynb) for detailed explanations and usage instructions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GeoLarryLai/pyTopoComplexity/blob/main/Example_pyrugosity.ipynb)

<p align="center">
 <img src="image/rugosity.png" width="100%" height="100%""/>
</p>

### 4. Terrain Position Index Calculation

```python
from pytopocomplexity import TPI
```

The module **pytpi.py** calculates the Terrain Position Index (TPI) of the land surface following ([Weiss, 2001](https://www.jennessent.com/arcview/TPI_Weiss_poster.htm)), which is a measure of the relative topographic position of a point in relation to the surrounding landforms. This metric is useful for determining surface ruggedness, classifying terrain, assessing local hydrodynamics, and identifying habitat hotspots. TPI, also known as the Topographic Position Index in terrestrial studies, distinguishes landscape features such as hilltops, valleys, flat plains, and upper or lower slopes. In oceanography, researchers adapt the Bathymetric Position Index (BPI), which applies the equivalent TPI algorithm to bathymetric data to assess seafloor complexity. Positive TPI values indicate locations that are higher than the average of their surroundings (e.g., ridges), while negative values indicate locations that are lower (e.g., valleys). Values near zero indicate flat areas or areas of constant slope. The module also returns the absolute values of the TPI, which only indicate the magnitude of the vertical position at each grid point relative to its neighbors.

See [**Example_pytpi.ipynb**](https://github.com/GeoLarryLai/pyTopoComplexity/blob/main/Example_pytpi.ipynb) for detailed explanations and usage instructions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GeoLarryLai/pyTopoComplexity/blob/main/Example_pytpi.ipynb)

<p align="center">
 <img src="image/tpi.png" width="100%" height="100%""/>
</p>

## Combinding pyTopoComplexity with Landscape Evolution Modeling

The Jupyter Notebook file [**Landlab_simulation.ipynb**](https://github.com/GeoLarryLai/pyTopoComplexity/blob/main/example/nonlineardiff_Landlab.ipynb) demonstrates the use of [Landlab](https://landlab.readthedocs.io/en/latest/index.html), an open-source Python framework for simulating landscape evolution and modeling time-dependent changes in topographic complexity driven by hillslope and fluvial processes. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GeoLarryLai/pyTopoComplexity/blob/main/Landlab_simulation.ipynb)

This notebook specifically employs two components from **Landlab**: the [`TaylorNonLinearDiffuser`](https://landlab.readthedocs.io/en/latest/generated/api/landlab.components.taylor_nonlinear_hillslope_flux.taylor_nonlinear_hillslope_flux.html#landlab.components.taylor_nonlinear_hillslope_flux.taylor_nonlinear_hillslope_flux.TaylorNonLinearDiffuser) from the [`terrainBento`](https://github.com/TerrainBento/terrainbento) package, developed by [Barnhart et al. (2019)](https://gmd.copernicus.org/articles/12/1267/2019/), which simulates topographic smoothing over time through nonlinear hillslope diffusion processes caused by near-surface soil disturbance and downslope soil creeping ([Roering et al., 1999](https://doi.org/10.1029/1998WR900090)), and the [`StreamPowerEroder`](https://landlab.readthedocs.io/en/latest/generated/api/landlab.components.stream_power.stream_power.html#landlab.components.stream_power.stream_power.StreamPowerEroder), a core component in **Landlab** that simulates topographic dissection through fluvial incision over time, following the method described by [Braun & Willett (2013)](https://doi.org/10.1016/j.geomorph.2012.10.008).

The notebook also includes functions that utilize modules from **pyTopoComplexity** to perform topographic complexity analysis on the simulated landscape from **Landlab** at each modeling timestep. The resulting GeoTIFF rasters, figures, and animations provide users with insights into the time-dependent changes in topographic complexity caused by land-surface processes.

<p align="center">
<img src="image/Landlab_demo.gif" width="100%" height="100%" align="center"/>
</p>

## Example DEM Raster Files

This repository include example LiDAR DEM files under `~/ExampleDEM/` that cover the area and nearby region of a deep-seated landslide occurred in 2014 at Oso area of the North Fork Stillaguamish River (NFSR) valley, Washington State, USA. The souce LiDAR DEM files was cropped from the 'Stillaguamish 2014' project that was originally contracted by Washington State Department of Transportation (WSDOT), downloaded from the [Washington Lidar Portal](http://lidarportal.dnr.wa.gov) on April 4, 2024. A goal of this work allow users to reproduce the research by [Booth et al. (2017)](https://doi.org/10.1002/2016JF003934) and permit comparison of topographic complexity metrics derived from other regions using **pyTopoComplexity** package and the [**Landlab_simulation.ipynb**](https://github.com/GeoLarryLai/pyTopoComplexity/blob/main/Landlab_simulation.ipynb) tool.

The example DEM raster files have various grid size, coordinate reference system (CRS), and unit of grid value (elevation, Z).

| LiDAR DEM Files  | CRS | XY Grid Size | Z Unit | Descriptions |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Ososlid2014_f_3ftgrid.tif | NAD83/Washington South (ftUS) (EPSG: 2286) | 3.0 [US survey feet] | US survey feet | 2014 Oso Landslide |
| Ososlid2014_m_3ftgrid.tif | NAD83/Washington South (EPSG: 32149) | ~0.9144 [meters] | meters | 2014 Oso Landslide |
| Ososlid2014_f_6ftgrid.tif | NAD83/Washington South (ftUS) (EPSG: 2286) | 6.0 [US survey feet] | US survey feet | 2014 Oso Landslide |
| Ososlid2014_m_6ftgrid.tif | NAD83/Washington South (EPSG: 32149) | ~1.8288 [meters] | meters | 2014 Oso Landslide |

> [!NOTE]
> When testing the code with the example DEM files, users should place the entire `~/ExampleDEM/` subfolder in the same directory as the Jupyter Notebook files. Both the **pyTopoComplexity** package and the [**Landlab_simulation.ipynb**](https://github.com/GeoLarryLai/pyTopoComplexity/blob/main/Landlab_simulation.ipynb) landscape evolution modeling tool have the capability to automatically detect the grid spacing and the units of the XYZ directions (must be in feet or meters) of the input DEM raster and compute the results in SI units.

## Requirements
For **pyTopoComplexity** package"
* Python >= 3.10
* `numpy` >= 1.20
* `scipy` >= 1.5
* `rasterio` >= 1.2
* `dask` >= 2021.0
* `matplotlib` >= 3.3
* `tqdm` >= 4.0
* `numba` >= 0.50
* `statsmodels` >= 0.12

Additional packages for Jupyter Notebook pyTopoComplexity examples:
* `pandas`  >= 2.1
* `jupyter` >= 1.0
* `IPython` >= 7.0
* `imageio` >= 2.9
* `gdal` >= 3.0

For landscape smoothing simulation:
* [`landlab`](https://landlab.readthedocs.io/en/latest/index.html) >= 2.7
  * Used modeling components: `TaylorNonLinearDiffuser`, `FlowAccumulator`, `StreamPowerEroder`
  * Used data processing and visualization components: `esri_ascii`, `imshowhs`, `imshowhs_grid`

See also the `environment.yml` file which can be used to create a virtual environment.

## License
**pyTopoComplexity** is licensed under the [Apache License 2.0](LICENSE).
