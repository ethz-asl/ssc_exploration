.PHONY: all build test clean

all: build
	CC=g++ LDSHARED='$(shell python3 scripts/configure.py)' python3 setup.py build
	

build:
ifeq (, $(shell which nvcc))
	@echo using CPU
	gcc -c voxel_util.cpp -o voxel_util.o -fopenmp
	rm -f libvoxelutil.a
	ar crs libvoxelutil.a voxel_util.o 
	ranlib libvoxelutil.a
	rm  voxel_util.o
else
	@echo using GPU
	nvcc -rdc=true --compiler-options '-fPIC' -c -o temp.o voxel_util.cu
	nvcc -dlink --compiler-options '-fPIC' -o voxel_util.o temp.o -lcudart
	rm -f libvoxelutil.a
	ar cru libvoxelutil.a voxel_util.o temp.o
	ranlib libvoxelutil.a
	rm temp.o voxel_util.o
	
endif

clean:
	rm -f libvoxelutil.a *.o main temp.py
	rm -rf build

