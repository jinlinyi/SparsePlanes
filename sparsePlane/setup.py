#!/usr/bin/env python3
from setuptools import find_packages, setup

setup(
    name="sparseplane",
    version="1.0",
    author="Linyi Jin, Shengyi Qian, Andrew Owens, David Fouhey",
    description="Code for Planar Surface Reconstruction From Sparse Views",
    packages=find_packages(exclude=("configs", "tests")),
    install_requires=["torchvision>=0.4", "fvcore", "detectron2", "pytorch3d"],
)
