# Setup 

This repository makes use of several external libraries and requires Python 3.7.  
We highly recommend installing them within a virtual environment such as pyenv or Anaconda. 

The script below will help you set up the environment:

1. Create a conda environment.
```bash 
# if you prefer pyenv
VERSION_ALIAS="sparseplane"  pyenv install anaconda3-2019.10
pyenv local sparseplane
pip install --upgrade pip

# if you prefer conda virtual environment
# make sure you use python 3.7 since KMSolver requires it.
conda create -n sparseplane python=3.7
pip install Cython  # in case you miss Cython
```

2. Install dependencies including pytorch

```bash
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
```

3. Install [Detectron2][1]
```
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
```
4. Install Sparse Plane
```
cd sparsePlane
pip install -e .
```


[1]: https://detectron2.readthedocs.io/en/latest/tutorials/install.html#install-pre-built-detectron2-linux-only
