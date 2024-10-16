from setuptools import setup, find_packages

with open("README-pypi.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pytopocomplexity",
    version="1.0.0",
    author="Larry Syu-Heng Lai",
    author_email="larrysyuhenglai@gmail.com",
    description="A package for topographic complexity analysis", 
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
        "scipy>=1.6",
        "rasterio>=1.2",
        "dask>=2021.0",
        "matplotlib>=3.3",
        "tqdm>=4.0",
        "numba>=0.53",
        "statsmodels>=0.12",
        "gdal>=3.9",
        "ipython>=8.14",
        "imageio>=2.11"
    ],
    license="Apache License 2.0",
)