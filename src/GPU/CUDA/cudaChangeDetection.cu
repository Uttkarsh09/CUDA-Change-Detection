#include "../../../include/GPU/CUDA/cudaChangeDetection.cuh"
#include "../../../include/GPU/CUDA/changeDetectionKernel.cuh"

#define THREADS_PER_BLOCK 1024

// Variable Declarations
Pixel *h_oldImagePixArr, *h_newImagePixArr, *h_highlightedChangePixArr;
Pixel *d_oldImagePixArr, *d_newImagePixArr, *d_highlightedChangePixArr;
int gpuChoice = -1;

void printCUDADeviceProperties(void)
{
	// Code
	cout << endl << "Detected Nvidia GPU ... Using CUDA ...";
	cout << endl << "--------------------------------------------------------------------------------------------------" << endl;
	cout << endl << "CUDA INFORMATION : " << endl;
	cout << endl << "**************************************************************************************************";
	
	cudaError_t retCudaRt;
	int devCount;

	retCudaRt = cudaGetDeviceCount(&devCount);

	if (retCudaRt != cudaSuccess)
	{
		cout << endl << "CUDA Runtime API Error - cudaGetDeviceCount() Failed Due To " << cudaGetErrorString(retCudaRt) << endl;
	}
	else if (devCount == 0)
	{
		cout << endl << "No CUDA Supported Devices Found On This System ... Exiting !!!" << endl;
		return;
	}
	else
	{
		for (int i = 0; i < devCount; i++)
		{
			cudaDeviceProp devProp;
			int driverVersion = 0, runtimeVersion = 0;

			retCudaRt = cudaGetDeviceProperties(&devProp, i);
			if (retCudaRt != cudaSuccess)
			{
				cout << endl << " " << cudaGetErrorString(retCudaRt) << "in" << __FILE__ << "at line " << __LINE__ << endl;
				return;
			}

			cudaDriverGetVersion(&driverVersion);
			cudaRuntimeGetVersion(&runtimeVersion);

			cout << endl << "GPU Device Number			: " << i;
			cout << endl << "GPU Device Name				: " << devProp.name;
			cout << endl << "GPU Device Memory			: " << (ceil((float)devProp.totalGlobalMem / 1048576.0f) / 1024.0f) << " GB";
			cout << endl << "GPU Device Number Of SMProcessors	: " << devProp.multiProcessorCount;
		}

		// GPU Selection
		if (devCount > 1)
		{
			cout << endl << "You have more than 1 CUDA GPU Devices ... Please select 1 of them";
			cout << endl << "Enter GPU Device Number : ";
			cin >> gpuChoice;

			// Set CUDA GPU Device
			cudaSetDevice(gpuChoice);
		}
		else
		{
			// Set CUDA GPU Device
			cudaSetDevice(0);
		}

		cout << endl << "**************************************************************************************************";
		cout << endl << "--------------------------------------------------------------------------------------------------" << endl;
	}
}


void runOnGPU(ImageData *oldImage, ImageData *newImage, int threshold, uint8_t *detectedChanges)
{
	size_t size = (oldImage->height * oldImage->pitch)/3;
	float timeOnGPU = 0.0f;

	h_oldImagePixArr = (Pixel*)malloc(size * sizeof(Pixel));
	h_newImagePixArr = (Pixel*)malloc(size * sizeof(Pixel));
	h_highlightedChangePixArr = (Pixel*)malloc(size * sizeof(Pixel));

	convertBitmapToPixelArr(h_oldImagePixArr, oldImage->bitmap, size);
	convertBitmapToPixelArr(h_newImagePixArr, newImage->bitmap, size);

	cudaMalloc(&d_oldImagePixArr, size * sizeof(Pixel));
	cudaMalloc(&d_newImagePixArr, size * sizeof(Pixel));
	cudaMalloc(&d_highlightedChangePixArr, size * sizeof(Pixel));

	printCUDADeviceProperties();

	cudaMemcpy(d_oldImagePixArr, h_oldImagePixArr, size * sizeof(Pixel), cudaMemcpyHostToDevice);
	cudaMemcpy(d_newImagePixArr, h_newImagePixArr, size * sizeof(Pixel), cudaMemcpyHostToDevice);

	dim3 blocks((size + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK);
	
	StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	detectChanges<<<blocks, THREADS_PER_BLOCK>>>(d_oldImagePixArr, d_newImagePixArr, d_highlightedChangePixArr, threshold, size);
	
	sdkStopTimer(&timer);
	timeOnGPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
 
	cudaMemcpy(h_highlightedChangePixArr, d_highlightedChangePixArr, size * sizeof(Pixel), cudaMemcpyDeviceToHost);

	cout << endl << "Time Taken on GPU : " << timeOnGPU << " ms" << endl;

	convertPixelArrToBitmap(detectedChanges, h_highlightedChangePixArr, size, false);

	cleanup();
}

void cleanup(void)
{
	if (d_highlightedChangePixArr)
	{
		cudaFree(d_highlightedChangePixArr);
		d_highlightedChangePixArr = NULL;
	}

	if (d_newImagePixArr)
	{
		cudaFree(d_newImagePixArr);
		d_newImagePixArr = NULL;
	}

	if (d_oldImagePixArr)
	{
		cudaFree(d_oldImagePixArr);
		d_oldImagePixArr = NULL;
	}

	if (h_highlightedChangePixArr)
	{
		free(h_highlightedChangePixArr);
		h_highlightedChangePixArr = NULL;
	}

	if (h_newImagePixArr)
	{
		free(h_newImagePixArr);
		h_newImagePixArr = NULL;
	}

	if (h_oldImagePixArr)
	{
		free(h_oldImagePixArr);
		h_oldImagePixArr = NULL;
	}
}
