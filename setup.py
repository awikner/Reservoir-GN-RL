#define NPY_NO_DEPRACATED_API NPY_1_7_API_VERSION
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='RLS functions',
    ext_modules=cythonize("RLS_functions_2.pyx"),
    include_dirs = [np.get_include()],
    zip_safe=False,
)