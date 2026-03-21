"""
build_tracker.py
================
Builds tracker_cy.pyx → tracker_cy.pyd  WITHOUT distutils, setuptools, or MSVC.
Works on Windows with MinGW-w64 (gcc) and on Linux/macOS with gcc/clang.

Usage
-----
  python build_tracker.py

Requirements
------------
  pip install cython numpy
  Windows: MinGW-w64 with gcc.exe on PATH
    → easiest install: https://winlibs.com  (UCRT, x86_64, latest .zip)
      extract to e.g. C:\\mingw64, add C:\\mingw64\\bin to PATH
    OR via Chocolatey (PowerShell as Admin):
      choco install mingw
"""

import os
import platform
import shutil
import struct
import subprocess
import sys
import sysconfig
import tempfile

# ── helpers ───────────────────────────────────────────────────────────────────
def run(cmd: list[str], *, env=None) -> None:
    print("  $", " ".join(cmd))
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        print(f"\nERROR: command exited with code {r.returncode}")
        sys.exit(r.returncode)


def find_exe(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        return ""
    return path


# ── dependency check ──────────────────────────────────────────────────────────
try:
    import numpy as np
    from Cython.Compiler.Main import compile as cython_compile
    from Cython.Compiler import Options as CyOptions
except ImportError:
    print("ERROR: run  pip install cython numpy  first.")
    sys.exit(1)

IS_WINDOWS = platform.system() == "Windows"
BITS = struct.calcsize("P") * 8          # 32 or 64

# ── locate gcc ────────────────────────────────────────────────────────────────
GCC = find_exe("gcc")
if not GCC:
    if IS_WINDOWS:
        print(
            "\nERROR: gcc not found on PATH.\n\n"
            "Install MinGW-w64 (pick ONE):\n\n"
            "  A) winlibs.com  →  UCRT x86_64 latest  →  extract, add <path>\\bin to PATH\n"
            "  B) Chocolatey (PowerShell as Admin):  choco install mingw\n"
            "  C) MSYS2 already installed:  pacman -S mingw-w64-x86_64-gcc\n"
            "     then add  C:\\msys64\\mingw64\\bin  to PATH\n\n"
            "Then re-run:  python build_tracker.py\n"
        )
    else:
        print("\nERROR: gcc not found. Install with:  sudo apt install gcc  (or equivalent)")
    sys.exit(1)

print(f"gcc   : {GCC}")
print(f"Python: {sys.executable}  ({sys.version.split()[0]})")
print(f"numpy : {np.__version__}")

# ── paths ─────────────────────────────────────────────────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
PYX_FILE   = os.path.join(HERE, "tracker_cy.pyx")
C_FILE     = os.path.join(HERE, "tracker_cy.c")

# Python extension suffix:  .cpython-312-x86_64-linux-gnu.so  or  .cp312-win_amd64.pyd
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or (
    ".pyd" if IS_WINDOWS else ".so"
)
OUT_FILE   = os.path.join(HERE, f"tracker_cy{EXT_SUFFIX}")

if not os.path.exists(PYX_FILE):
    print(f"ERROR: {PYX_FILE} not found — make sure tracker_cy.pyx is in the same folder.")
    sys.exit(1)

# ── Step 1: Cython  .pyx → .c ────────────────────────────────────────────────
print("\n[1/3] Cythonizing tracker_cy.pyx → tracker_cy.c")
CyOptions.docstrings         = False
CyOptions.annotate           = True    # produces tracker_cy.html (yellow=slow lines)

from Cython.Build import cythonize
from Cython.Compiler.Main import CompilationOptions, default_options, compile

opts = dict(default_options)
opts.update({
    "language_level":      3,
    "output_file":         C_FILE,
    "annotate":            True,
    "compiler_directives": {
        "boundscheck":  False,
        "wraparound":   False,
        "cdivision":    True,
        "nonecheck":    False,
        "embedsignature": False,
    },
})

result = compile(PYX_FILE, full_module_name="tracker_cy", **opts)
if result.num_errors > 0:
    print("Cython compilation failed.")
    sys.exit(1)

if not os.path.exists(C_FILE):
    print(f"ERROR: expected {C_FILE} was not produced by Cython.")
    sys.exit(1)
print("  OK")

# ── Step 2: compile .c → .o ───────────────────────────────────────────────────
print("\n[2/3] Compiling tracker_cy.c")

py_include = sysconfig.get_path("include")           # Python headers
np_include = np.get_include()                        # NumPy headers

OBJ_FILE = C_FILE.replace(".c", ".o")

cc_flags = [
    "-O3", "-march=native", "-ffast-math",
    "-fPIC",                                         # position-independent (Linux)
    "-c",                                            # compile only, don't link yet
    C_FILE, "-o", OBJ_FILE,
    f"-I{py_include}",
    f"-I{np_include}",
    "-DPY_ARRAY_UNIQUE_SYMBOL=tracker_cy_ARRAY_API",
    "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
]

if IS_WINDOWS:
    cc_flags += [
        "-DMS_WIN64",
        "-D_hypot=hypot",                            # MinGW quirk with Python headers
        "-Wno-unused-result",
    ]

run([GCC] + cc_flags)
print("  OK")

# ── Step 3: link .o → .pyd / .so ─────────────────────────────────────────────
print(f"\n[3/3] Linking → {os.path.basename(OUT_FILE)}")

# Python library to link against
py_libdir  = sysconfig.get_config_var("LIBDIR") or ""
py_version = f"{sys.version_info.major}{sys.version_info.minor}"

if IS_WINDOWS:
    # On Windows the import lib lives inside the Python install
    py_lib_path = os.path.join(os.path.dirname(sys.executable), "libs")
    py_lib_name = f"python{py_version}"              # e.g. python312

    ld_flags = [
        "-shared", OBJ_FILE, "-o", OUT_FILE,
        f"-L{py_lib_path}", f"-l{py_lib_name}",
        "-static-libgcc",                            # bundle libgcc, no DLL dep
        "-Wl,--enable-auto-import",
    ]
else:
    # Linux / macOS: don't link libpython — let the interpreter provide it
    ld_flags = [
        "-shared", OBJ_FILE, "-o", OUT_FILE,
        "-Wl,-rpath," + (py_libdir or ""),
    ]
    if platform.system() == "Darwin":
        ld_flags += ["-undefined", "dynamic_lookup"]

run([GCC] + ld_flags)

# ── Done ──────────────────────────────────────────────────────────────────────
if os.path.exists(OUT_FILE):
    size_kb = os.path.getsize(OUT_FILE) // 1024
    print(f"\n✓ Built successfully: {OUT_FILE}  ({size_kb} KB)")
    print("\nVerifying import …")
    result = subprocess.run(
        [sys.executable, "-c",
         "import tracker_cy; print('  tracker_cy loaded OK — Cython IoU active')"],
        cwd=HERE,
    )
    if result.returncode != 0:
        print("  WARNING: import test failed — check the error above.")
else:
    print("\nERROR: output file was not created.")
    sys.exit(1)
