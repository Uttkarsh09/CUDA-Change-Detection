#!/bin/bash

clear
stty -echo

cd ./bin/

GPU_INFO=`nvcc --version | grep NVIDIA`

if [ "$GPU_INFO" == "" ]; then
	printf "OpenCL Underprogress...\n"
else
	nvcc -I ../include/ -c ../src/GPU/CUDA/*.cu 

	nvcc  -I ../include/ -c ../src/main.cpp ../src/common/*.cpp ../src/CPU/*.cpp
	
	nvcc -o ../detectChanges -L ../lib/ *.o -lfreeimage
fi

cd ..
./detectChanges
stty echo
