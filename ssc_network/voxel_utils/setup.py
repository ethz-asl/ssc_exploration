import os
from distutils.core import setup, Extension
import numpy as np

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

if 'CUDA_PATH' in os.environ:
   CUDA_PATH = os.environ['CUDA_PATH']
else:
   print("Could not find CUDA_PATH in environment variables. Defaulting to /usr/local/cuda!")
   CUDA_PATH = "/usr/local/cuda"

if not os.path.isdir(CUDA_PATH):
   print("CUDA_PATH {} not found. Switching to CPU!")
   setup(name = 'VoxelUtils', version = '1.0',  \
      ext_modules = [
         Extension('VoxelUtils', ['voxel_util_module.c'], 
         include_dirs=[np.get_include()],
         libraries=["voxelutil"],
         extra_link_args = ["-fopenmp"],
         library_dirs = ["."]
   )])
else:
   setup(name = 'VoxelUtils', version = '1.0',  \
      ext_modules = [
         Extension('VoxelUtils', ['voxel_util_module.c'], 
         include_dirs=[np.get_include(), os.path.join(CUDA_PATH, "include")],
         libraries=["voxelutil", "cudart"],
         library_dirs = [".", os.path.join(CUDA_PATH, "lib64")]
   )])
