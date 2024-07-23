from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pytopocomplexity",
    version="0.7.1",
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
        "os",
        "numpy",
        "rasterio",
        "matplotlib",
        "dask",
        "numba",
        "scipy",
        "tqdm",
    ],
)
