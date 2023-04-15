#include "headers.hpp""

void listDevices(){
	int deviceCount=-1;
	int c;
	int *dev_c;
	cudaDeviceProp deviceProperties;

	cudaGetDeviceCount(&deviceCount);
	
	for(int i=0 ; i<deviceCount ; i++){
		cudaGetDeviceProperties(&deviceProperties, i);
		printDeviceInformation(deviceProperties);
		printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n");
	}
}
