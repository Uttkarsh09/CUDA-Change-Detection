#include "../../../include/GPU/CUDA/cudaChangeDetection.cuh"
#include "../../../include/GPU/CUDA/changeDetectionKernel.cuh"
#define THREADS_PER_BLOCK 1024

// Variable Declarations
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

void convertBitmapToPixelArr(uint8_t *bitmap, Pixel *pixelArr, size_t size)
{
	for(int i=0 ; i<size ; i++, bitmap+=3)
	{
		pixelArr[i].blue = bitmap[0];
		pixelArr[i].green = bitmap[1];
		pixelArr[i].red = bitmap[2];
	}
}


void convertPixelArrToBitmap(Pixel *pixelArr, uint8_t *bitmap, size_t size){
	for(int i=0 ; i<size ; i++, bitmap+=3)
	{
		bitmap[0] = pixelArr[i].blue;
		bitmap[1] = pixelArr[i].green;
		bitmap[2] = pixelArr[i].red;
	}
}


void runOnGPU(ImageData *oldImage, ImageData *newImage, int threshold, uint8_t *detectedChanges)
{
	Pixel *h_oldImagePixArr, *h_newImagePixArr, *h_highlightedChangesPixArr;
	Pixel *d_oldImagePixArr, *d_newImagePixArr, *d_highlightedChangesPixArr;
	size_t size = (oldImage->height * oldImage->pitch)/3;
	uint8_t *bitmapPtrCpy;

	h_oldImagePixArr = (Pixel*)malloc(size * sizeof(Pixel));
	h_newImagePixArr = (Pixel*)malloc(size * sizeof(Pixel));
	h_highlightedChangesPixArr = (Pixel*)malloc(size * sizeof(Pixel));

	bitmapPtrCpy = oldImage->bitmap;
	convertBitmapToPixelArr(bitmapPtrCpy, h_oldImagePixArr, size);
	bitmapPtrCpy = newImage->bitmap;
	convertBitmapToPixelArr(bitmapPtrCpy, h_newImagePixArr, size);

	cudaMalloc(&d_oldImagePixArr, size * sizeof(Pixel));
	cudaMalloc(&d_newImagePixArr, size * sizeof(Pixel));
	cudaMalloc(&d_highlightedChangesPixArr, size * sizeof(Pixel));

	// printCUDADeviceProperties();

	cudaMemcpy(d_oldImagePixArr, h_oldImagePixArr, size * sizeof(Pixel), cudaMemcpyHostToDevice);
	cudaMemcpy(d_newImagePixArr, h_newImagePixArr, size * sizeof(Pixel), cudaMemcpyHostToDevice);

	dim3 blocks((size + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK);
	
	auto start = std::chrono::high_resolution_clock::now();
	detectChanges<<<blocks, THREADS_PER_BLOCK>>>(d_oldImagePixArr, d_newImagePixArr, d_highlightedChangesPixArr, threshold, size);
	auto stop = std::chrono::high_resolution_clock::now();

	auto GPU_Duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	cout << "GPU Duration = " << GPU_Duration.count() << endl;

	cudaMemcpy(h_highlightedChangesPixArr, d_highlightedChangesPixArr, size * sizeof(Pixel), cudaMemcpyDeviceToHost);

	bitmapPtrCpy = detectedChanges;
	convertPixelArrToBitmap(h_highlightedChangesPixArr, bitmapPtrCpy, size);

	free(h_oldImagePixArr);
	free(h_newImagePixArr);
	free(h_highlightedChangesPixArr);
	cudaFree(d_highlightedChangesPixArr);
	cudaFree(d_oldImagePixArr);
	cudaFree(d_newImagePixArr);
}


void cleanup(void)
{
	cout << endl << "Placeholder Cleanup Message" << endl;
}
