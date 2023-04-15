#!/bin/bash

compileCPU(){
	cd ./bin/CPU/
	g++ -I ../../include/ ../../src/CPU/*.cpp -o detectChanges -L ../../ -lfreeimage
	cd ../../
}

compileCUDA(){
	printf "CUDA Underprogress...\n"
}

compileOpenCL(){
	printf "OpenCL Underprogress...\n"
}

echo "COMPILING ON CPU"
compileCPU

printf "\nChecking for GPU\n"
GPU_INFO=`nvcc --version | grep NVIDIA`

if [ "$GPU_INFO" == "" ]; then
	echo "NVIDIA GPU NOT AVAILABLE"
	echo "USING OpenCL Instead"
	compileOpenCL
else
	echo "NVIDIA GPU AVAILABLE"
	echo "USING CUDA"
	compileCUDA
fi

