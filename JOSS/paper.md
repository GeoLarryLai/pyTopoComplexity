---
title: 'pyTopoComplexity: A Python package for topographic complexity analysis'
tags:
  - Python
  - topographic complexity
  - surface roughness
  - two-dimensional continuous wavelet transform
  - fractal dimension
  - rugosity index
authors:
  - name: Larry Syu-Heng Lai
    corresponding: true
    orcid: 0000-0003-4589-9729 
    affiliation: "1"
  - name: Adam M. Booth
    orcid: 0000-0002-7339-0594   
    affiliation: "2"
  - name: Alison R. Duvall
    orcid: 0000-0002-7760-7236 
    affiliation: "1"
  - name: Erich Herzig
    orcid: 0000-0002-3898-5971 
    affiliation: "1"
affiliations:
 - name: Department of Earth and Space Sciences, University of Washington, Seattle, Washington, USA
   index: 1
 - name: Department of Geology, Portland State University, Portland, Oregon, USA
   index: 2
date: 31 August 2024
bibliography: paper.bib
---

# Summary

**pyTopoComplexity** is a Python package that provides a computationally efficient and highly 
customizable implementation of three methods for quantifying topographic complexity. 
These methods include two-dimensional continuous wavelet 
transform (2D-CWT) analysis, fractal dimension estimation, and rugosity index calculation across 
various spatial scales. This package addresses the scarcity of open-source software for these 
sophisticated methods, which are crucial in modern terrain analysis, and facilitates data 
comparison and reproducibility. In the [software respository](https://github.com/GeoLarryLai/pyTopoComlexity.git), 
we also include a Jupyter Notebook file that integrates components from the Python-based 
surface-process modeling platform **Landlab** [@Hobley2017]. This allows researchers to 
simulate the smoothing of topography over time through terrestrial nonlinear hillslope 
diffusion processes. By combining these features, **pyTopoComplexity** advances the toolset 
available to researchers for measuring and simulating the time-dependent persistence of 
topographic complexity signatures against environmental forces on terrain surfaces.

# Statement of need

Topographic complexity, often referred to as topographic roughness or surface roughness, provides 
critical insights into surface processes and the interactions among the geosphere, biosphere, and 
hydrosphere. With the increasing availability, utility, and popularity of digital terrain model (DTM) 
data, quantifying topographic complexity has become an essential measure in terrain analysis across 
various research fields. This necessity spans applications such as terrain classification and mapping 
at various spatial scales [@Weiss2001; @Robbins2018; @Lindsay2019; @PardoIguzquiza2022a], evaluating 
the depositional age of event sedimentation and subsequent erosion processes 
[@Hetz2016; @Johnstone2018; @Booth2017; @LaHusen2020; @Herzig2023], and identifying habitats to assess 
ecological diversity on land and seafloor [@Frost2005; @Hetz2016; @Wilson2007].

In recent years, several advanced methods for quantifying topographic complexity have been developed, 
including two-dimensional continuous wavelet transform (2D-CWT) analysis [@Booth2009; @Berti2013], 
fractal dimension estimation [@Taud2005; @Robbins2018; @PardoIguzquiza2020], and rugosity index 
calculation [@Jenness2004; @DuPreez2015]. These methods are considered more effective for terrain analysis 
tasks compared to conventional morphological metrics such as variations in local slope and relief. 
Despite their importance, comprehensive publicly available tools that incorporate these advanced 
methods for studying topographic complexity are lacking. Common open-source geospatial analysis software, 
such as QGIS [@QGIS_software], GRASS GIS [@GRASS_GIS_software], and WhiteboxTools [@Lindsay2016], only 
implement basic conventional methods, limiting the reproducibility and comparability of these newer 
approaches. Although some specialized programs for calculating the rugosity index exist [@Walbridge2018; @Benham2022], 
they have been limited to marine bathymetric studies and involve various mathematical choices and designs.

To address this gap, we have developed an open-source Python toolkit called **pyTopoComplexity**, which provides 
computationally efficient and easily customizable implementations of three modules for performing and 
visualizing the results of 2D-CWT, fractal dimension, and rugosity calculations (Table 1\autoref{tab:1}). 
This toolkit can detect the grid spacing and unit of the projected coordinate system (acceptable in meters, 
U.S. survey feet, and international feet) from the input raster DTM file (GeoTIFF format) and 
automatically conduct unit conversions in necessary calculation steps to ensure data consistency and 
reproducibility. Results at nodes affected by edge effects due to no-data values outside the input raster 
will be removed by default. Users can define the suitable spatial scale to match their research purposes 
and choose computational approaches (e.g., chunk processing, faster mathematical approximations) to 
optimize performance (see details in the **Methods and features overview** section).

| Modules  | Classes | Method Descriptions | References |
| ------------- | ------------- | ------------- | ------------- |
| pycwtmexhat.py | CWTMexHat | Quantifies the wavelet-based curvature of the land surface using two-dimensional continuous wavelet transform (2D-CWT) with a Mexican Hat wavelet | @Booth2009; @Booth2017 |
| pyfracd.py | FracD | Conducts fractal dimension analysis on the land surface using variogram procedures | @Wen1997; @PardoIguzquiza2020|
| pyrugostiy.py | RugosityIndex | Calculates the rugosity index of the land surface | @Jenness2004; @DuPreez2015 |
Table: Table 1 \label{tab:1}: Modules contained in the **pyTopoComplexity** package.

Each module of the **pyTopoComplexity** is provided with a corresponding [example Jupyter Notebook file](https://github.com/GeoLarryLai/pyTopoComlexity/tree/main/example) 
for usage instructions, using the Light Detection and Ranging (LiDAR) DTM data of a deep-seated landslide that 
occurred in 2014 in the Oso area of the North Fork Stillaguamish River valley, Washington State, USA 
[@WashingtonGeologicalSurvey2023]. In the software repository, we also include an additional Jupyter Notebook file 
[**nonlineardiff_Landlab.ipynb**](https://github.com/GeoLarryLai/pyTopoComlexity/blob/main/example/nonlineardiff_Landlab.ipynb), 
which allows researchers to simulate the smoothing of topography over time via terrestrial nonlinear hillslope 
diffusion processes [@Roering1999]. This is achieved by employing the `TaylorNonLinearDiffuser` module from the 
`terrainBento` Python package [@Barnhart2019] and running the simulation in the **Landlab** environment [@Hobley2017].

By bridging the gap between different advanced terrain analytical approaches and incorporating functionality 
for landscape evolution modeling, **pyTopoComplexity** serves as a valuable resource for topographic complexity 
research and has the potential to foster new insights and interdisciplinary collaborations in the fields 
of geology, geomorphology, geography, ecology, and oceanography.

# Methods and features overview

## Two-dimensional continuous wavelet transform analysis

The **pycwtmexhat.py** module in **pyTopoComplexity** implements the 2D-CWT method for terrain analysis, providing detailed information on how 
amplitude is distributed across spatial frequencies at each position in the data by transforming spatial data into position-frequency 
space. This method is particularly effective for depicting the Laplacian of topography [@Torrence1998; @Lashermes2007], revealing 
concave and convex regions of topography at various scales [@Malamud2001; @Struble2021], identifying deep-seated landslides 
[@Booth2009; @Berti2013], and estimating the depositional ages of landslide deposits [@Booth2017; @LaHusen2020; @Underwood2022; @Herzig2023].

The 2D-CWT is computed by convolving the elevation data $z$ with a wavelet family $\psi$, using a wavelet scale parameter $s$ at every location ($x$, $y$):

$$
C (s, x, y) = \Delta^2 \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} z(x, y) \psi \left( x, y \right) dx \, dy
$$

, where the resultant wavelet coefficient $C(s,x,y)$ provides a measure of how well the wavelet $\psi$ matches the data $z$ at each grid [@Torrence1998]. 
When $s$ is large, $\psi$ is spread out, capturing long-wavelength features of $z$; when $s$ is small, $\psi$ becomes more localized, making it sensitive 
to fine-scale features of $z$. In this implementation, we use the 2D Mexican Hat wavelet (i.e., Ricker wavelet) function to define $\psi$:

$$
\psi = − \frac{1}{\pi(s\Delta)^4}(1-\frac{𝑥^2+𝑦^2}{2s^2})e^{(-\frac{𝑥^2+𝑦^2}{2s^2})}
\:\:\:\:\:\:\:\:\:
\lambda=\frac{2\pi s}{\sqrt{5/2}}\Delta
$$

The Mexican hat wavelet is proportional to the second derivative of a Gaussian envelope, with its Fourier wavelength ($\lambda$) 
ependent on the grid spacing ($\Delta$) of the input DTM raster. The wavelet function $\psi$ is scaled according to the 
wavelet scale parameter $s$ and the grid spacing $\Delta$, ensuring that the resultant wavelet coefficient $C$ signifies 
concave and convex landforms corresponding to the wavelet scale $s$. Users can define the value of $\lambda$ in meters as the 
targeted spatial scale for landform roughness analysis, and the **pycwtmexhat.py** module will automatically compute the wavelet 
scale $s$ based on the grid spacing ($\Delta$) of the input raster file (Figure 1)\autoref{fig:1}.

We note that the $C$ and $\psi$ equations presented here are the mathmetical approaches adapted in later publications 
[@LaHusen2020; @Underwood2022; @Herzig2023] and ongoing works [@Booth2023; @Lai2023; @Ozioko2023] of landslide mapping and 
age dating studies, with minor differences from the earlier similar research in @Booth2009 and @Booth2017 (original MATLAB codes 
available from [Booth's personal website](https://web.pdx.edu/~boothad/tools.html)). These differences in methatical approach 
can result $C$ values in different units and order of magnitude (e.g., $10^{-3}$ to $10^{-4}$ [m$^{-2}$] in @Booth2017 
and prior studies; $10^{-2}$ to $10^{-3}$ [m$^{-1}$] in @LaHusen2020 thereafter). Despite this discrepency, the $C$ values 
yielded from these two approaches are linearly scaled and interconvertible, and they both reflect identical spatiotemporal patterns 
of topographic complexity (i.e., surface roughness). 

![Figure 1. Hillshade map of the 2014 Oso Landslide region (upper-left) and the results of the two-dimensional continuous wavelet 
transform (2D-CWT) analysis using the **pycwtmexhat.py** module, with designated Fourier wavelengths ($\lambda$) of the Mexican Hat 
wavelet at 15 m, 30 m, 45 m, 60 m, and 75 m. \label{fig:1}](fig1-pycwtmexhat.png){ width=100% }

## Fractal dimension analysis

The **pyfracd.py** module in **pyTopoComplexity** calculates the fractal dimension, which measures the fractal characteristics of natural 
features [@Mandelbrot1983]. This method provides insights into the self-similarity of landscapes, helping quantify their irregularity 
and fragmentation, which is crucial for studying the surface processes that shape Earth and planetary surfaces [@Xu1883].

In this module, we adapt the variogram method to estimate the local fractal dimension within a moving window centered at each cell of 
the DTM [@Taud2005; @PardoIguzquiza2020; @PardoIguzquiza2022a]. This approach simplifies the problem to estimating the fractal dimension 
of one-dimensional topographic profiles [@Dubuc1989] within the two-dimensional moving window. For a one-dimensional profile of length $L$, 
the variogram $\gamma_1(k)$ can be estimated at the $K$ lag distances ($K$ lag distances ($k = 1, \ldots, K$) by:

$$
\gamma_1(k) = \frac{1}{2(L-k)} \sum_{l=1}^{L-k} [Z(i) - Z(i+l)]^2
$$

, where $Z(i)$ is the elevation at location $i$ along the profile. The local fractal dimension ($FD$) is estimated from one-dimensional 
profiles in principal directions (i.e., horizontal, vertical, and diagonal) within a square moving window. Assuming that fractional 
Brownian motion is an appropriate stochastic model for natural surfaces, its variogram follows a power-law model with respect to $k$ [@Wen1997]:

$$
\gamma_1(k) = \alpha h^\beta, \quad \alpha \geq 0; \quad 0 \leq \beta < 2
$$

, and its exponent $\beta$ is related to the local fractal dimension ($FD$) by:

$$
FD = TD + 1 - \frac{\beta}{2}
$$

, where $TD$ is the topological dimension in the Euclidean space of the fractional Brownian motion. For one-dimensional fractional Brownian 
motion, $TD = 1$; thus, the fractal dimension of the two-dimensional surface $(FD)_2^*$ can be estimated as the average fractal dimension of 
the one-dimensional profiles $(FD)_1^*$:

$$
(FD)_2^* = 1 + (FD)_1^*
$$

Users can specify the size (number of grids along each edge) of the moving window to study fractal characteristics at desired spatial scales.
In addition to calculating the fractal dimension, the **pyfracd.py** module also computes reliability parameters such as standard error and 
the coefficient of determination ($R^2$) to assess the robustness of the analysis (Figure 2)\autoref{fig:2}.

![Figure 2. Hillshade map of the 2014 Oso Landslide region (upper-left), example variograms (right), and the results of Fractal Dimension 
(FD) analysis using the **pyfracd.py** module. The analysis was conducted with a designated 17 by 17 grid moving window size 
(grid spacing = 3 U.S. survey feet ≈ 0.9144 meters). \label{fig:2}](fig2-pyfracd.png){ width=100% }



## Rugosity index calculation

The **pyrugosity.py** module in **pyTopoComplexity**  measures the rugosity index of the land surface, which is widely used to assess structural 
complexity of the topography. Such method has been applied in classifying seafloor types by marine geologists and geomorphologist, small-scale 
hydrodynamics by oceanographers, and studying available habitats in the landscape by ecologists and coral biologists [@Lundblad2006; @Wilson2007].

The rugosity index is determined as the ratio of the contoured area (i.e., true geometric surface area) to the planimetric area within the square 
moving window, highlighting smaller-scale variations in surface height: 

$$
\text{Rugosity Index} = \frac{\text{contoured area}}{\text{planimetric area}}
$$

This module adapts the Triangulated Irregular Networks (TIN) method from @Jenness2004 to approximate the contoured area as the sum of 
eight truncated-triangle areas. These triangles connect the central grids, four corner grids, and four grids at the middle points of the 
surrounding edges within the moving window. If no local slope correction is applied, the planimetric area is considered to be the horizontal 
planar area of the moving window, as described in @Jenness2004. Another approach considers slope correction where to the planimetric area is 
projected onto an plane of the local gradient [@DuPreez2015].

By definition, the rugosity index is as a minimum value of one (completely flate surface). Typical valuesrange of the conventional rugosity 
index (without slope correction) from one to three although larger values are possible in very steep terrains. The slope-corrected rugosity 
index, also called arc-chord ratio (ACR) rugosity index, could provide a better representation of local surface complexity (Figure 3)\autoref{fig:3}.

![Figure 3. Hillshade map of the 2014 Oso Landslide region (left), along with the calculated results of the Arc-Chord Ratio Rugosity Index (middle) 
and the conventional Rugosity Index (right), using the **pyrugosity.py** module. The calculations were performed with a designated 17 by 17 grid 
moving window size (grid spacing = 3 U.S. survey feet ≈ 0.9144 meters). \label{fig:3}](fig3-pyrugosity.png){ width=100% }



## Forward simulation of landscape smoothing through nonlinear hillslope diffusion process

The `nonlineardiff_Landlab.ipynb` notebook in the **pyTopoComplexity** package offers a sophisticated tool for simulating landscape evolution 
through nonlinear diffusion processes due to near-surface soil disturbances and downslope sediment creep [@Roering1999]. This tool runs the 
simulation in the **Landlab** environment (version >= 2.7) [@Hobley2017] with the `TaylorNonLinearDiffuser` module from the `terrainBento` 
Python package [@Barnhart2019]. The main simulation iteratively applies the nonlinear diffusion model to predict changes in surface 
elevation $z$ over time $t$:

$$
\frac{\partial z}{\partial t} = -\nabla \cdot \mathbf{q}_s
$$

, where \(\mathbf{q}_s\) represents the sediment flux at the surface. The sediment flux is further defined by a nonlinear flux law that is 
approximated using a Taylor series expansion [@Ganti2012]:

$$
\mathbf{q}_s = K \mathbf{S} \left[1 + \sum_{i=1}^N \left( \frac{S}{S_c}\right)^{2i}\right]
$$

Here, $\mathbf{S} = -\nabla z$ represents the downslope topographic gradient, and $S$ is its magnitude. The parameter $K$ is a diffusion-like 
transport coefficient with dimensions of length squared per time. The simulation also incorporates the critical slope gradient ($S_c$) to 
ensure numerical stability and prevent the numerical instability when $S = S_c$. $N$ denotes the number of terms in the Taylor 
expansion, while $i$ specifies the number of additional terms included. If $N = 1$, the expression simplifies to linear diffusion [@Culling1963]. 
The default is set to $N = 2$ that gives the behavior described in @Ganti2012 as an approximation of the nonlinear diffusion.

This notebook provides a comprehensive workflow that guides users through setting up, importing raster files, and running simulations. Since 
**Landlab** primarily handles DTM data in ESRI ASCII format, this notebook includes utility functions for converting raster files between GeoTIFF 
and ESRI ASCII formats. Users are required to specify the values for $S_c$, $K$, the length of each time step in years, and the final time to 
stop the simulation. The example included in the notebook uses LiDAR DTM data from the 2014 Oso Landslide [@WashingtonGeologicalSurvey2023], 
with parameter values provided in @Booth2017, in an attempt to reproduce the simulation results presented in that study (Figure 4\autoref{fig:4}).

![Figure 4. Hillshade map of the 2014 Oso Lanslide region and surface smoothing evolution over 15,000 years predicted by a 
nonlinear hillslope diffusion model used in the `pyrugosity` module, in attempt to reproduce the simulation results in 
@Booth2017.\label{fig:3}](fig4-smoothing.png){ width=100% }

# Acknowledgements

The development of **pyTopoComplexity** is a part of collactive effort within the [Landslide subteam](https://cascadiacopeshub.org/team-1-landslides/) of the 
[Cascadia Coastlines and Peoples Hazards Research Hub (Cascadia CoPes Hub)](https://cascadiacopeshub.org), which is supported by funds from the 
National Science Foundation Award ([#2103713](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2103713&HistoricalAwards=false)). We express special thanks to 
Dr. Eulogio Pardo-Igúzquiza who shared his Fortran code for fractal dimension analysis used in his work @PardoIguzquiza2022b, which greatly inspired the development 
of **pyfracd.py** module.

The development of **pyTopoComplexity** is part of a collaborative effort within the [Landslide subteam](https://cascadiacopeshub.org/team-1-landslides/) of the 
[Cascadia Coastlines and Peoples Hazards Research Hub (Cascadia CoPes Hub)](https://cascadiacopeshub.org), funded by the National Science Foundation Award ([#2103713](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2103713&HistoricalAwards=false)). 
We extend special thanks to Dr. Eulogio Pardo-Igúzquiza, who generously shared his Fortran code for fractal dimension analysis used in his work 
@PardoIguzquiza2022b, which significantly inspired the development of the **pyfracd.py** module.

# References