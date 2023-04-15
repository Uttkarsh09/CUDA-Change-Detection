// #include<iostream>
// #include "FreeImage.h"

// __global__ void add(int a, int b, int *c){
// 	*c = a + b;
// }

// int main(){
// 	int deviceCount=-1;
// 	int c;
// 	int *dev_c;
// 	cudaDeviceProp deviceProperties;

// 	cudaGetDeviceCount(&deviceCount);
// 	for(int i=0 ; i<deviceCount ; i++){
// 		cudaGetDeviceProperties(&deviceProperties, i);
// 		printMachineInformation(deviceProperties);
// 		printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n");
// 	}


// 	cudaMalloc((void**)&dev_c, sizeof(int));
// 	add<<<1,1>>>(2, 7, dev_c);
// 	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
// 	printf("answer = %d", c);
// 	cudaFree(dev_c);
// 	return 0;
// }
