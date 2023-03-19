#include<iostream>

void printMachineInformation(cudaDeviceProp prop){
	int count;

	cudaGetDeviceCount(&count);
	for(int i=0 ; i<count ; i++){
		cudaGetDeviceProperties(&prop, i);
	}

	printf("~~~ CUDA INFORMATION ~~~\n");
	printf("Number of GPUs - %d\n", count);
	printf("Name of GPU - %s\n", prop.name);
	printf("Major Compute Capability - %d\n", prop.major);
	printf("Minor Compute Capability - %d\n", prop.minor);
	printf("Total Global Memory - %zu\n", prop.totalGlobalMem);
	printf("Shared Memory per Block - %zu\n", prop.sharedMemPerBlock);
}