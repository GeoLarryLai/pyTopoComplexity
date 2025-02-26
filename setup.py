from setuptools import setup, find_packages

with open("README-pypi.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pytopocomplexity",
    version="1.1.1",
    author="Larry Syu-Heng Lai",
    author_email="larrysyuhenglai@gmail.com",
    description="A package for multiscale topographic complexity analysis", 
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GeoLarryLai/pyTopoComplexity",
    packages=find_packages(exclude=['image', 'image.*', 'examples', 'examples.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.5",
        "rasterio>=1.2",
        "dask>=2021.0",
        "matplotlib>=3.3",
        "tqdm>=4.0",
        "numba>=0.50",
        "statsmodels>=0.12",
        "gdal>=3.0",
        "ipython>=7.0",
        "imageio>=2.9"
    ],
    license="Apache License 2.0",
)