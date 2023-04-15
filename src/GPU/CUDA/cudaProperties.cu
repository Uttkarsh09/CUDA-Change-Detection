#include "../../../include/GPU/CUDA/cudaProperties.cuh"

void getPlatformInfo(void)
{
	printCUDADeviceProperties();	
}

void printCUDADeviceProperties(void)
{
	// Code
	cout << endl << "Detected Nvidia GPU ... Using CUDA ...";
	cout << endl << "--------------------------------------------------------------------------------------------------" << endl;
	cout << endl << "CUDA INFORMATION : " << endl;
	cout << endl << "**************************************************************************************************";
	
	cudaError_t ret_cuda_rt;
	int dev_count;

	ret_cuda_rt = cudaGetDeviceCount(&dev_count);

	if (ret_cuda_rt != cudaSuccess)
	{
		cout << endl << "CUDA Runtime API Error - cudaGetDeviceCount() Failed Due To " << cudaGetErrorString(ret_cuda_rt) << endl;
	}
	else if (dev_count == 0)
	{
		cout << endl << "No CUDA Supported Devices Found On This System ... Exiting !!!" << endl;
		return;
	}
	else
	{
		for (int i = 0; i < dev_count; i++)
		{
			cudaDeviceProp dev_prop;
			int driverVersion = 0, runtimeVersion = 0;

			ret_cuda_rt = cudaGetDeviceProperties(&dev_prop, i);
			if (ret_cuda_rt != cudaSuccess)
			{
				cout << endl << " " << cudaGetErrorString(ret_cuda_rt) << "in" << __FILE__ << "at line " << __LINE__ << endl;
				return;
			}

			cout << endl;

			cudaDriverGetVersion(&driverVersion);
			cudaRuntimeGetVersion(&runtimeVersion);

			cout << "GPU Device Number			: " << i;
			cout << endl << "GPU Device Name				: " << dev_prop.name;
			cout << endl << "GPU Device Memory			: " << (ceil((float)dev_prop.totalGlobalMem / 1048576.0f) / 1024.0f) << " GB";
			cout << endl << "GPU Device Number Of SMProcessors	: " << dev_prop.multiProcessorCount;
		}

		cout << endl << "**************************************************************************************************" << endl;
		cout << endl << "--------------------------------------------------------------------------------------------------" << endl;
	}
}
