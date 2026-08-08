from setuptools import setup, find_namespace_packages

NAME = "sparta-sw"
VERSION = "0.0.1"
DESCRIPTION = ""

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author="Akis Nousias, Panos Roditis",
    author_email="nousiasx@gmail.com, pkroditis@gmail.com",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    entry_points={
        "console_scripts": [
            "train-vit = bin.train_vit:main",
            "infer-vit = bin.infer_vit:main",
        ]
    },
    install_requires=[
        "torch>=2.0.0,<2.13",
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.4.0",
        "torchvision",
        "opencv-python>=4.8.0",
        "toml>=0.10",
        "datasets==4.7.0",
        "wandb>=0.17.0",
        "entmax==1.3",
        "einops==0.8.2",
        "matplotlib>=3.10.9",
        "plotly==6.7.0",
        "brevitas==0.13.0",
        "onnx==1.22.0",
        "brevitas-utils@git+https://github.com/V0XNIHILI/brevitas-utils.git",
    ],
    dependency_links=[
        "https://download.pytorch.org/whl/cpu",
    ],
)
