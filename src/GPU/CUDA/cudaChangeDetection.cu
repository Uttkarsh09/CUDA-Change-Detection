#include "../../../include/GPU/CUDA/cudaChangeDetection.cuh"

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

		cout << endl << "**************************************************************************************************" << endl;
		cout << endl << "--------------------------------------------------------------------------------------------------" << endl;
	}
}

void runOnGPU(void)
{
	printCUDADeviceProperties();

	cleanup();
}

void cleanup(void)
{
	cout << endl << "Placeholder Cleanup Message" << endl;
}
