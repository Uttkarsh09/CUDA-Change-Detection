#include "../../../include/GPU/OpenCL/openclProperties.hpp"

void getPlatformInfo(void)
{
	printOpenCLDeviceProperties();
}

void printOpenCLDeviceProperties(void)
{
	// Variable Declarations
	cl_int result;
	cl_platform_id ocl_platform_id;
	cl_uint dev_count, compute_units;
	cl_ulong mem_size;
	cl_bool image_support;
	cl_device_id* ocl_device_ids;
	char oclPlatformInfo[512], ocl_dev_prop[1024];

	// Code
	cout << endl << "Detected AMD/Intel/Nvidia GPU ... Using OpenCL ...";
	cout << endl << "--------------------------------------------------------------------------------------------------" << endl;
	cout << endl << "OpenCL INFORMATION : " << endl;
	cout << endl << "**************************************************************************************************" << endl;
	
	// Get 1st Platform ID
	result = clGetPlatformIDs(1, &ocl_platform_id, NULL);	
	
	if (result != CL_SUCCESS)
	{
		cout << endl << "clGetPlatformIDs() Failed ... " << endl;
		exit(EXIT_FAILURE);
	}

	// Get GPU Device Count
	result = clGetDeviceIDs(ocl_platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &dev_count);
	if (result != CL_SUCCESS)
	{
		cout << endl << "clGetDeviceIDs() Failed ... " << endl;
		exit(EXIT_FAILURE);
	}
	else if (dev_count == 0)
	{
		cout << endl << "No OpenCL Supported Device Found On This System !!!" << endl;
		exit(EXIT_FAILURE);
	}
	else
	{
		// Allocate Memory To Hold Those Device Ids
		ocl_device_ids = (cl_device_id*)malloc(sizeof(cl_device_id) * dev_count);

		// Get Ids Into Allocated Buffer
		clGetDeviceIDs(ocl_platform_id, CL_DEVICE_TYPE_GPU, dev_count, ocl_device_ids, NULL);

		for (int i = 0; i < (int)dev_count; i++)
		{
			cout << "GPU Device Number				: " << i;
			
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_NAME, sizeof(ocl_dev_prop), &ocl_dev_prop, NULL);
			cout << endl << "GPU Device Name					: " << ocl_dev_prop;
			
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_VENDOR, sizeof(ocl_dev_prop), &ocl_dev_prop, NULL);
			cout << endl << "GPU Device Vendor				: " << ocl_dev_prop;

			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_VERSION, sizeof(ocl_dev_prop), &ocl_dev_prop, NULL);
			cout << endl << "GPU Device OpenCL Version			: " << ocl_dev_prop;
			
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
			cout << endl << "GPU Device Memory				: " << (unsigned long long) mem_size / 1000000000 << " GB";
			
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
			cout << endl << "GPU Device Number Of Parallel Processor Cores	: " << compute_units;
			
			clGetDeviceInfo(ocl_device_ids[i], CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
			if (image_support == CL_TRUE)
				cout << endl << "GPU Device Image Support			: Yes";
			else
				cout << endl << "GPU Device Image Support			: No";
		}

		free(ocl_device_ids);

		cout << endl << "**************************************************************************************************" << endl;
		cout << endl << "--------------------------------------------------------------------------------------------------" << endl;
	}
}
