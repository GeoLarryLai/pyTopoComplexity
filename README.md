# pyTopoComlexity (v.1.1)

### Warning: These codes still under development and not yet officially published. Additional tests are required.

This repository includes a set of codes that perform various types of geomorphic metrics representing topographic complexity and/or variability. The geomorphic metrics included here are:  
* _Mexican Hat Wavelet Tranform Analysis_ (pyMexicanHat.ipynb)
* _Terrain Ruggedness Index_ (pyTRI.ipynb)
* _Rugosity_ (pyRugosity.ipynb)
* _Roughness_ (pyRoughness.ipynb)
* _Fractal Dimension_ (pyFractalD.ipynb)

I also include the code that calculate the surface anomalies, showing where are peaks and troughs.
* _Bathymetric Position Index_ (pyBPI.ipynb)

Recommended reference for these topographic metrics:  
  * Wilson, M.F.J., O’Connell, B., Brown, C., Guinan, J.C., Grehan, A.J., 2007. Multiscale Terrain Analysis of Multibeam Bathymetry Data for Habitat Mapping on the Continental Slope. Marine Geodesy 30, 3-35. https://doi.org/10.1080/01490410701295962
  * Du Preez, C. A new arc–chord ratio (ACR) rugosity index for quantifying three-dimensional landscape structural complexity. Landscape Ecol 30, 181–192 (2015). https://doi.org/10.1007/s10980-014-0118-8

======   
The Mexican Hat Wavelet Tranform code (writen in MATLAB) was originally developed from Dr. Adam M. Booth (Portland State Univeristy). Citations:
  * Booth, A.M., Roering, J.J., Perron, J.T., 2009. Automated landslide mapping using spectral analysis and high-resolution topographic data: Puget Sound lowlands, Washington, and Portland Hills, Oregon.     Geomorphology 109, 132-147. https://doi.org/10.1016/j.geomorph.2009.02.027
  *Booth, A.M., LaHusen, S.R., Duvall, A.R., Montgomery, D.R., 2017. Holocene history of deep-seated landsliding in the North Fork Stillaguamish River valley from surface roughness analysis, radiocarbon dating, and numerical landscape evolution modeling. Journal of Geophysical Research: Earth Surface 122, 456-472. https://doi.org/10.1002/2016JF003934

This MATLAB code was later adapted by Dr. Sean R. LaHusen (Univeristy of Washington) & revised by Erich N. Herzig (Univeristy of Washington).  Citations:
  * LaHusen, S.R., Duvall, A.R., Booth, A.M., Montgomery, D.R., 2016. Surface roughness dating of long-runout landslides near Oso, Washington (USA), reveals persistent postglacial hillslope instability. Geology 44, 111-114. https://doi.org/10.1130/G37267.1
  * LaHusen, S.R., Duvall, A.R., Booth, A.M., Grant, A., Mishkin, B.A., Montgomery, D.R., Struble, W., Roering, J.J., Wartman, J., 2020. Rainfall triggers more deep-seated landslides than Cascadia earthquakes in the Oregon Coast Range, USA. Science Advances 6, eaba6790. https://doi.org/10.1126/sciadv.aba6790
  * Herzig, E.N., Duvall, A.R., Booth, A.R., Stone, I., Wirth, E., LaHusen, S.R., Wartman, J., Grant, A.; Evidence of Seattle Fault Earthquakes from Patterns in Deep‐Seated Landslides. Bulletin of the Seismological Society of America 2023; https://doi.org/10.1785/0120230079

In November, 2023, this MATLAB code was optimized by Dr. Larry Syu-Heng Lai (Univeristy of Washington). It is further translated into a python version. Citations:
 * Lai, L. S.-H., Booth, A.M., Herzig, E.N., LaHusen, S.R., & Alison, D. (2023). pyTopoComplexity (1.1). Zenodo. https://doi.org/10.5281/zenodo.10080534
