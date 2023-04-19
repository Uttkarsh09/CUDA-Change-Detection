#include "../../../include/GPU/OpenCL/openclChangeDetection.hpp"

// Variable Declarations
cl_platform_id oclPlatformId;
cl_device_id oclDeviceId;
cl_uint devCount, computeUnits;
cl_ulong memorySize;
cl_bool imageSupport;
cl_device_id* oclDeviceIds;
char oclPlatformInfo[512], oclDevProp[1024];

cl_context oclContext;
cl_command_queue oclCommandQueue;

cl_program oclProgram;
cl_kernel oclKernel;

cl_int result;

int gpuChoice = -1;

void getOpenCLPlatforms(void)
{
    result = clGetPlatformIDs(1, &oclPlatformId, NULL);
	if (result != CL_SUCCESS)
	{
		cout << endl << "clGetPlatformIDs() Failed : " << result << endl;
		cleanup();
        exit(EXIT_FAILURE);
	}
}

void getOpenCLDevices(void)
{
	result = clGetDeviceIDs(oclPlatformId, CL_DEVICE_TYPE_GPU, 0, NULL, &devCount);
	if (result != CL_SUCCESS)
	{
		cout << endl << "clGetDeviceIDs() Failed : " << result << endl;
		cleanup();
        exit(EXIT_FAILURE);
	}
	else if (devCount == 0)
	{
		cout << endl << "No OpenCL Supported Device Found On This System !!!" << endl;
		exit(EXIT_FAILURE);
	}
	else
	{
		// Allocate Memory To Hold Those Device Ids
		oclDeviceIds = (cl_device_id*)malloc(sizeof(cl_device_id) * devCount);

		// Get Ids Into Allocated Buffer
		clGetDeviceIDs(oclPlatformId, CL_DEVICE_TYPE_GPU, devCount, oclDeviceIds, NULL);

		for (int i = 0; i < (int)devCount; i++)
		{
			cout << "GPU Device Number				: " << i;
			
			clGetDeviceInfo(oclDeviceIds[i], CL_DEVICE_NAME, sizeof(oclDevProp), &oclDevProp, NULL);
			cout << endl << "GPU Device Name					: " << oclDevProp;
			
			clGetDeviceInfo(oclDeviceIds[i], CL_DEVICE_VENDOR, sizeof(oclDevProp), &oclDevProp, NULL);
			cout << endl << "GPU Device Vendor				: " << oclDevProp;

			clGetDeviceInfo(oclDeviceIds[i], CL_DEVICE_VERSION, sizeof(oclDevProp), &oclDevProp, NULL);
			cout << endl << "GPU Device OpenCL Version			: " << oclDevProp;
			
			clGetDeviceInfo(oclDeviceIds[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memorySize), &memorySize, NULL);
			cout << endl << "GPU Device Memory				: " << (unsigned long long) memorySize / 1000000000 << " GB";
			
			clGetDeviceInfo(oclDeviceIds[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, NULL);
			cout << endl << "GPU Device Number Of Parallel Processor Cores	: " << computeUnits;
			
			clGetDeviceInfo(oclDeviceIds[i], CL_DEVICE_IMAGE_SUPPORT, sizeof(imageSupport), &imageSupport, NULL);
			if (imageSupport == CL_TRUE)
				cout << endl << "GPU Device Image Support			: Yes";
			else
				cout << endl << "GPU Device Image Support			: No";
		}

		// GPU Selection
		if (devCount > 1)
		{
			cout << endl << "You have more than 1 OpenCL GPU Devices ... Please select 1 of them";
			cout << endl << "Enter GPU Device Number : ";
			cin >> gpuChoice;

			// Set OpenCL GPU Device
			oclDeviceId = oclDeviceIds[gpuChoice];
		}
		else
		{
			// Set OpenCL GPU Device
			oclDeviceId = oclDeviceIds[0];
		}
	}
}

void printOpenCLDeviceProperties(void)
{
	// Code
	cout << endl << "Detected AMD/Intel/Nvidia GPU ... Using OpenCL ...";
	cout << endl << "--------------------------------------------------------------------------------------------------" << endl;
	cout << endl << "OpenCL INFORMATION : " << endl;
	cout << endl << "**************************************************************************************************" << endl;

	getOpenCLPlatforms();

	getOpenCLDevices();

	cout << endl << "**************************************************************************************************" << endl;
	cout << endl << "--------------------------------------------------------------------------------------------------" << endl;
}

void createOpenCLContext(void)
{
    oclContext = clCreateContext(NULL, 1, &oclDeviceId, NULL, NULL, &result);
	if (result != CL_SUCCESS)
	{
		cout << endl << "clCreateContext() Failed : " << result << endl;
		cleanup();
        exit(EXIT_FAILURE);
	}
}

void createOpenCLCommandQueue(void)
{
	oclCommandQueue = clCreateCommandQueue(oclContext, oclDeviceId, 0, &result);
	if (result != CL_SUCCESS)
	{
		cout << endl << "clCreateCommandQueue() Failed : " << result << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}
}

void runOnCPU(ImageData *oldImage, ImageData *newImage, int threshold, uint8_t *detectedChanges)
{
	// ** Does the job of getting OpenCLPlatformID and and OpenCLDeviceID
	printOpenCLDeviceProperties();

	createOpenCLContext();

	createOpenCLCommandQueue();

	cleanup();
}

void cleanup(void)
{
    cout << endl << "Placeholder Cleanup Message" << endl;

	if (oclDeviceIds)
	{
		free(oclDeviceIds);
	}
}
