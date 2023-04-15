#!/bin/bash

# ? Moving into the bin directory
cd ./bin/

compileCPU(){
	cd ./CPU/
	g++  -I ../../include/ -c ../../src/CPU/*.cpp
	cd ../
}

compileCUDA(){
	cd ./GPU/CUDA/
	nvcc -I ../../../../include/ -c ../../../*.cu 
	cd ../../
}

compileOpenCL(){
	cd ./GPU/OpenCL
	printf "OpenCL Underprogress...\n"
	cd ../../
}

compileCPU

GPU_INFO=`nvcc --version | grep NVIDIA`

if [ "$GPU_INFO" == "" ]; then
	compileOpenCL
else
	compileCUDA
	g++ -o detectChanges -L ../../lib/ ./CPU/*.o ./GPU/CUDA/*.o -lfreeimage
fi

cd ..
