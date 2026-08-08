# sparta-sw

A repository for the software activities of the SPARTA project. The software activities of the SPARTA 
project are supported by `sparse_vit`, a python package.


## Installation

The software activities of the SPARTA project are supported by `sparse_vit`, a python package that can be
installed as-is in a virtual environment such as `conda` or `venv`.


### Installation as packages

Preferably, create a new python environment to hold the package installations. Make sure that the new environment
includes basic installation libraries such as `wheel` and `pip`. Usually, these are supported by default for new 
`conda` and `venv` environments. 

Navigate [the package folder](./sparse_vit/) and build a package `.whl` file:
```
python setup.py bdist_wheel 
```

Then install the packages via their `.whl` files using `pip`:
```
pip install <my-package-name>.whl
```

### Installation in development mode

Alternativelly, it can be installed in editable mode by navigating to [the package folder](./sparse_vit/)
and execute an editable installation as follows:
```
pip install -e ./src
```
