"""
Build 2048 C++ accelerated engine.
Usage:
    python setup.py build_ext --inplace
    python setup.py build_ext --inplace --compiler=mingw32
"""
from setuptools import setup, Extension
import sys
import os
import subprocess

try:
    import pybind11
    pybind11_include = pybind11.get_include()
except ImportError:
    print("Please install pybind11: pip install pybind11")
    sys.exit(1)

# Auto-detect compiler and set flags
def get_compiler():
    """Detect whether we're using MSVC or GCC-style compiler."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import sysconfig; print(sysconfig.get_platform())"],
            capture_output=True, text=True
        )
    except Exception:
        pass

    # Check sys.argv for --compiler flag
    for i, arg in enumerate(sys.argv):
        if arg == "--compiler" and i + 1 < len(sys.argv):
            if "mingw" in sys.argv[i + 1]:
                return "mingw"
        if "mingw" in arg:
            return "mingw"

    # Default: MSVC on Windows, GCC elsewhere
    if sys.platform == "win32":
        return "msvc"
    return "gcc"

compiler = get_compiler()

if compiler == "mingw":
    extra_compile_args = ["-O3", "-march=native", "-ffast-math"]
    extra_link_args = ["-static-libgcc", "-static-libstdc++", "-Wl,-Bstatic", "-lstdc++", "-lpthread"]
elif sys.platform == "win32":
    extra_compile_args = ["/O2"]
    extra_link_args = []
else:
    extra_compile_args = ["-O3", "-march=native", "-fopenmp", "-ffast-math"]
    extra_link_args = ["-fopenmp"]

ext = Extension(
    "game2048_cpp",
    sources=["game2048_cpp.cpp"],
    include_dirs=[pybind11_include],
    language="c++",
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    define_macros=[("NDEBUG", "1")],
)

setup(
    name="game2048_cpp",
    version="1.0.0",
    description="2048 Game Engine (C++ Accelerated)",
    ext_modules=[ext],
    python_requires=">=3.8",
)
