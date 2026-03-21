# setup_tracker.py
# Build the Cython tracker extension:
#   python setup_tracker.py build_ext --inplace
#
# After building, a  tracker_cy*.so  (Linux/Mac) or  tracker_cy*.pyd  (Windows)
# appears in the same directory.  webcam_full_v4.py imports it automatically
# and falls back to pure-Python if it's absent.

import sys
from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
    import numpy as np
except ImportError:
    print("ERROR: Cython and numpy are required to build the extension.")
    print("  pip install cython numpy")
    sys.exit(1)

ext = Extension(
    name="tracker_cy",
    sources=["tracker_cy.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=[
        "-O3",          # maximum optimisation
        "-march=native",# use all CPU features available on this machine
        "-ffast-math",  # allow FP re-association (safe for IoU)
    ],
    language="c",
)

setup(
    name="tracker_cy",
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            "language_level": "3",
            "boundscheck":    False,
            "wraparound":     False,
            "cdivision":      True,
            "nonecheck":      False,
        },
        annotate=True,   # generates tracker_cy.html — shows which lines are still Python
    ),
)
