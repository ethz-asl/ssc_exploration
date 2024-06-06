# VoxelUtils
A 3D Volumetric Voxel Processing library using CUDA and CPU. The library provides the following funcitonality:
* TSDF Calculation
* Calculating 3D Projection Volume from Depth Images
* 3D Grid based Volumetric Probabilistic Fusion

The backend is compiled as static library `libvoxelutil.a` . Python bindings are exported in `voxel_util_module.c` for enabling the C++/CUDA usage from Python.
> **_NOTE:_** First the availability of CUDA compiler is checked, which if found, is preferred otherwise the CPU backend code is compiled. 

### Installation
```bash script
cd voxel_utils
make
pyhton setup.py install
```
