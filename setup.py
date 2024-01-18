from distutils.core import setup
from setuptools import find_packages

setup(
    name="luminance_analysis",
    version="0.1",
    author="Luigi Petrucco & Ot Prat",
    author_email="lpetrucco@neuro.mpg.de",
    packages=find_packages(),
    install_requires=[
        "ipython",
        "numba",
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-image",
        "ipywidgets",
        "pandas",
        "flammkuchen",
        "seaborn"
    ],
)
