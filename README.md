# pyTopoComlexity (v0.4)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10909837.svg)](https://doi.org/10.5281/zenodo.10909837)
### In this latest update (v0.4), the Mexican Hat code is confirmed to be working correctly.
### Warning: For other codes (Rugosity, Fractal Dimension, Topographic Position Index), please use them with caution. These codes are under development and required further testing.

This repository includes a set of codes that perform various types of geomorphic metrics representing topographic complexity and/or variability. The geomorphic metrics included here are:  
* _Mexican Hat Wavelet Tranform Analysis_ (pyMexicanHat.ipynb)
* _Rugosity_ (pyRugosity.ipynb)
* _Fractal Dimension_ (pyFD.ipynb)
* _Topographic Position Index_ (pyTPI.ipynb)

Additional Jupyter Notebooks using chunk processing & incremetal writing approach are included, in case for avoiding memory crash while running Mexican Hat Wavelet Tranform Analysis on large GeoTIFFs (pyMexicanHat_chunk.ipynb; pyMexicanHat_chunkIC.ipynb).

Recommended reference for these topographic metrics:  
  * Wilson, M.F.J., O’Connell, B., Brown, C., Guinan, J.C., Grehan, A.J., 2007. Multiscale Terrain Analysis of Multibeam Bathymetry Data for Habitat Mapping on the Continental Slope. Marine Geodesy 30, 3-35. https://doi.org/10.1080/01490410701295962
  * Mark, D.M., Aronson, P.B., 1984. Scale-dependent fractal dimensions of topographic surfaces: An empirical investigation, with applications in geomorphology and computer mapping. Journal of the International Association for Mathematical Geology 16, 671-683. https://doi.org/10.1007/BF01033029
  * Taud, H., Parrot, J.-F., 2006. Measurement of DEM roughness using the local fractal dimension. Géomorphologie : relief, processus, environnement 4. https://doi.org/10.4000/geomorphologie.622
  * Weiss, A. D. 2001. Topographic Positions and Landforms Analysis (poster), ESRI International User Conference, July 2001. San Diego, CA: ESRI.
  * Lundblad, E., D. J. Wright, J. Miller, E. M. Larkin, R. Rinehart, D. F. Naar, B. T. Donahue, S. M. Anderson, and T. Battista. 2006. A Benthic Terrain Classification Scheme for American Samoa. Marine Geodesy 29(2):89–111. https://doi.org/10.1080/01490410600738021
  * A portion of the code in pyFD.ipynb was consulted with xDEM (https://xdem.readthedocs.io). Citation: https://doi.org/10.5281/zenodo.4809698
    
======   
The Mexican Hat Wavelet Tranform code was originally developed in MATLAB by Dr. Adam M. Booth (Portland State Univeristy). Citations:
  * Booth, A.M., Roering, J.J., Perron, J.T., 2009. Automated landslide mapping using spectral analysis and high-resolution topographic data: Puget Sound lowlands, Washington, and Portland Hills, Oregon.     Geomorphology 109, 132-147. https://doi.org/10.1016/j.geomorph.2009.02.027
  *Booth, A.M., LaHusen, S.R., Duvall, A.R., Montgomery, D.R., 2017. Holocene history of deep-seated landsliding in the North Fork Stillaguamish River valley from surface roughness analysis, radiocarbon dating, and numerical landscape evolution modeling. Journal of Geophysical Research: Earth Surface 122, 456-472. https://doi.org/10.1002/2016JF003934

This MATLAB code was later adapted by Dr. Sean R. LaHusen (Univeristy of Washington) & revised by Erich N. Herzig (Univeristy of Washington).  Citations:
  * LaHusen, S.R., Duvall, A.R., Booth, A.M., Montgomery, D.R., 2016. Surface roughness dating of long-runout landslides near Oso, Washington (USA), reveals persistent postglacial hillslope instability. Geology 44, 111-114. https://doi.org/10.1130/G37267.1
  * LaHusen, S.R., Duvall, A.R., Booth, A.M., Grant, A., Mishkin, B.A., Montgomery, D.R., Struble, W., Roering, J.J., Wartman, J., 2020. Rainfall triggers more deep-seated landslides than Cascadia earthquakes in the Oregon Coast Range, USA. Science Advances 6, eaba6790. https://doi.org/10.1126/sciadv.aba6790
  * Herzig, E.N., Duvall, A.R., Booth, A.R., Stone, I., Wirth, E., LaHusen, S.R., Wartman, J., Grant, A.; Evidence of Seattle Fault Earthquakes from Patterns in Deep‐Seated Landslides. Bulletin of the Seismological Society of America 2023; https://doi.org/10.1785/0120230079

In November, 2023, this MATLAB code was optimized by Dr. Larry Syu-Heng Lai (Univeristy of Washington) and further translated into the python version here. Citations:
 * Lai, L. S.-H. (2024). pyTopoComplexity (0.4). Zenodo. https://doi.org/10.5281/zenodo.10909837
