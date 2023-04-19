#!/bin/bash
set -e

clear

cd ./bin/

GPU_INFO=`nvcc --version | grep NVIDIA`

if [ "$GPU_INFO" == "" ]; then
	printf "OpenCL Underprogress...\n"
else
	nvcc -c ../src/GPU/CUDA/*.cu 

	nvcc -c ../src/main.cpp ../src/common/*.cpp ../src/CPU/*.cpp
	
	nvcc -o ../detectChanges -L ../lib/ *.o -lfreeimage
fi

cd ..
./detectChanges
