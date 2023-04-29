#include "../../../include/GPU/OpenCL/openclChangeDetection.hpp"
#include "../../../include/common/helper_timer.h"

// Variable Declarations
cl_platform_id oclPlatformId;
cl_device_id oclDeviceId;
cl_uint devCount, computeUnits, maxSamplers, maxReadImgArgs, maxWriteImgArgs;
cl_ulong memorySize;
cl_bool imageSupport;
cl_device_id* oclDeviceIds;
char oclPlatformInfo[512], oclDevProp[1024];

cl_context oclContext;
cl_command_queue oclCommandQueue;

cl_program oclProgram;
cl_kernel oclKernel;

cl_int result;

cl_image_format oclImageFormat;

int gpuChoice = -1;

Pixel *h_oldImagePixArr, *h_newImagePixArr, *h_highlightedChangePixArr;
cl_mem d_oldImage, d_newImage, d_highlightedChanges;

ofstream logFile;

const char* oclSourceCode =
	
	"__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;" \
	
	"__kernel void rgbToGrayScale(__read_only image2d_t inputOld, __read_only image2d_t inputNew, __write_only image2d_t output, int threshold)" \
	"{" \
		"int2 gid = (int2)(get_global_id(0), get_global_id(1));" \
		
		"uint4 oldPixel, newPixel, finalPixelColor;" \
		"uint oldGrayVal, newGrayVal, difference;" \
		
		"int2 sizeOld = get_image_dim(inputOld);" \
		"int2 sizeNew = get_image_dim(inputNew);" \

		"if (all(gid < sizeOld) && all(gid < sizeNew))" \
		"{" \
			"oldPixel = read_imageui(inputOld, sampler, gid);" \
			"oldGrayVal = (0.3 * oldPixel.x) + (0.59 * oldPixel.y) + (0.11 * oldPixel.z);" \
			"newPixel = read_imageui(inputNew, sampler, gid);" \
			"newGrayVal = (0.3 * newPixel.x) + (0.59 * newPixel.y) + (0.11 * newPixel.z);" \
		"}" \

		"difference = abs_diff(oldGrayVal, newGrayVal);" \

		"if (difference >= threshold)" \
		"{" \
			"finalPixelColor = (uint4)(0, 0, 255, 255);" \
		"}" \
		"else" \
		"{" \
			"finalPixelColor = (uint4)(oldGrayVal, oldGrayVal, oldGrayVal, 255);" \
		"}" \
		
		"write_imageui(output, gid, finalPixelColor);" \
	"}";

void getOpenCLPlatforms(void)
{
	// Code
    result = clGetPlatformIDs(1, &oclPlatformId, NULL);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clGetPlatformIDs() Failed : " << getErrorString(result) << " ... Exiting !!!" << endl;
		cleanup();
        exit(EXIT_FAILURE);
	}
}

void getOpenCLDevices(void)
{
	// Code
	result = clGetDeviceIDs(oclPlatformId, CL_DEVICE_TYPE_GPU, 0, NULL, &devCount);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clGetDeviceIDs() Failed : " << getErrorString(result) << " ... Exiting !!!" << endl;
		cleanup();
        exit(EXIT_FAILURE);
	}
	else if (devCount == 0)
	{
		cerr << endl << "No OpenCL Supported Device Found On This System " << getErrorString(result) << " ... Exiting !!!" << endl;
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

			clGetDeviceInfo(oclDeviceIds[i], CL_DEVICE_MAX_SAMPLERS, sizeof(maxSamplers), &maxSamplers, NULL);
			cout << endl << "GPU Device Max Samplers				: " << maxSamplers;

			clGetDeviceInfo(oclDeviceIds[i], CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(maxReadImgArgs), &maxReadImgArgs, NULL);
			cout << endl << "GPU Device Max Read Image Arguments		: " << maxReadImgArgs;

			clGetDeviceInfo(oclDeviceIds[i], CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(maxWriteImgArgs), &maxWriteImgArgs, NULL);
			cout << endl << "GPU Device Max Write Image Arguments		: " << maxWriteImgArgs;
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
	// Code
    oclContext = clCreateContext(NULL, 1, &oclDeviceId, NULL, NULL, &result);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clCreateContext() Failed : " << getErrorString(result) << " ... Exiting !!!" << endl;
		cleanup();
        exit(EXIT_FAILURE);
	}
}

void createOpenCLCommandQueue(void)
{
	// Code
	oclCommandQueue = clCreateCommandQueue(oclContext, oclDeviceId, 0, &result);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clCreateCommandQueue() Failed : " << getErrorString(result) << " ... Exiting !!!" << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}
}

void createOpenCLImageStructure(ImageData* oldImage, ImageData* newImage)
{
	// Code

	// Initialize cl_image_format structure
	oclImageFormat.image_channel_order = CL_RGBA;
	oclImageFormat.image_channel_data_type = CL_UNSIGNED_INT8;

	// Create Old Input Image
	d_oldImage = clCreateImage2D(oclContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &oclImageFormat, oldImage->width, oldImage->height, 0, h_oldImagePixArr, &result);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clCreateImage2D() Failed For Old Input Image : " << getErrorString(result) << " ... Exiting !!!" << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}

	// Create New Input Image
	d_newImage = clCreateImage2D(oclContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &oclImageFormat, newImage->width, newImage->height, 0, h_newImagePixArr, &result);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clCreateImage2D() Failed For New Input Image : " << getErrorString(result) << " ... Exiting !!!" << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}

	// Create Ouput Image (Highlighted Changes)
	d_highlightedChanges = clCreateImage2D(oclContext, CL_MEM_WRITE_ONLY, &oclImageFormat, newImage->width, newImage->height, 0, NULL, &result);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clCreateImage2D() Failed For Highlighted Changes : " << getErrorString(result) << " ... Exiting !!!" << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}
	
}

void createOpenCLProgram(const char *kernelFileName)
{
	// Variable Declarations
	ifstream kernelFile(kernelFileName, ios::in);
	ostringstream outputStringStream;
	string clSourceContents;
	const char* clSourceCharArray = NULL;

	// Code
	// // ** Read from .cl file buffer into outputStringStream
	// outputStringStream << kernelFile.rdbuf();

	// clSourceContents = outputStringStream.str();
	// clSourceCharArray = clSourceContents.c_str();

	//oclProgram = clCreateProgramWithSource(oclContext, 1, (const char**)&clSourceCharArray, NULL, &result);
	oclProgram = clCreateProgramWithSource(oclContext, 1, (const char**)&oclSourceCode, NULL, &result);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clCreateProgramWithSource() Failed : " << getErrorString(result) << " ... Exiting !!!" << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}

	// ** Build Program
	result = clBuildProgram(oclProgram, 0, NULL, NULL, NULL, NULL);
	if (result != CL_SUCCESS)
	{
		size_t logSize;
		char *log;

		clGetProgramBuildInfo(oclProgram, oclDeviceId, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

		log = (char*)malloc(logSize);
		if (log == NULL)
		{
			cerr << "Failed to create log" << getErrorString(result) << " ... Exiting !!!" << endl;
			cleanup();
			exit(EXIT_FAILURE);
		}

		clGetProgramBuildInfo(oclProgram, oclDeviceId, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);

		logFile << endl << "Program Build Log : " << log << endl;
		
		cerr << endl << "clBuildProgram() Failed : " << result << endl;

		clReleaseProgram(oclProgram);
		cleanup();
		exit(EXIT_FAILURE);
	}
}

void createOpenCLKernel(int threshold, int count)
{
	oclKernel = clCreateKernel(oclProgram, "oclChangeDetection", &result);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clCreateKernel() Failed : " << getErrorString(result) << " ... Exiting !!!" << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), &d_oldImage);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clSetKernelArg() Failed For 1st Argument : " << getErrorString(result) << " ... Exiting !!!" << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = clSetKernelArg(oclKernel, 1, sizeof(cl_mem), &d_newImage);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clSetKernelArg() Failed For 2nd Argument : " << getErrorString(result) << " ... Exiting !!!" << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = clSetKernelArg(oclKernel, 2, sizeof(cl_mem), &d_highlightedChanges);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clSetKernelArg() Failed For 3rd Argument : " << getErrorString(result) << " ... Exiting !!!" << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = clSetKernelArg(oclKernel, 3, sizeof(cl_int), (void*)&threshold);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clSetKernelArg() Failed For 4th Argument : " << getErrorString(result) << " ... Exiting !!!" << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}
}

size_t roundUp(int localSize, unsigned int globalSize)
{
	// Code
	unsigned int remainder = globalSize % localSize;

	if (remainder == 0)
		return globalSize;
	else
		return globalSize + localSize - remainder;
}

void runOnGPU(ImageData *oldImage, ImageData *newImage, int threshold, uint8_t *detectedChanges)
{
	// Code
	logFile.open("Log.txt");

	printOpenCLDeviceProperties();

	size_t size = (oldImage->height * oldImage->pitch)/3;
	float timeOnGPU = 0.0f;

	h_oldImagePixArr = (Pixel*)malloc(size * sizeof(Pixel));
	if (h_oldImagePixArr == NULL)
	{
		cout << endl << "Failed to allocate memory for h_oldImagePixArr ... Exiting !!!" << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}
	
	h_newImagePixArr = (Pixel*)malloc(size * sizeof(Pixel));
	if (h_newImagePixArr == NULL)
	{
		cout << endl << "Failed to allocate memory for h_newImagePixArr ... Exiting !!!" << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}
	
	h_highlightedChangePixArr = (Pixel*)malloc(size * sizeof(Pixel));
	if (h_highlightedChangePixArr == NULL)
	{
		cout << endl << "Failed to allocate memory for h_highlightedChangePixArr ... Exiting !!!" << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}

	convertBitmapToPixelArr(h_oldImagePixArr, oldImage->bitmap, size);
	convertBitmapToPixelArr(h_newImagePixArr, newImage->bitmap, size);

	createOpenCLContext();

	createOpenCLCommandQueue();

	createOpenCLImageStructure(oldImage, newImage);

	createOpenCLProgram("oclChangeDetection.cl");

	createOpenCLKernel(threshold);

	// ! ** IMPORTANT ***
	// ! CL_KERNEL_WORK_GROUP_SIZE <= CL_DEVICE_MAX_WORK_GROUP_SIZE

	// Kernel Configuration
	size_t localSize[2] = {32, 32};
	//size_t globalSize[2] = {imageWidth, imageHeight};
	size_t globalSize[2] = {(unsigned int)oldImage->width, (unsigned int)oldImage->height};

	// Start Timer
	StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	result = clEnqueueNDRangeKernel(oclCommandQueue, oclKernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
	if (result != CL_SUCCESS)
	{
		cout << endl << "clEnqueueNDRangeKernel() Failed : " << getErrorString(result) << " ... Exiting !!!" << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}

	clFinish(oclCommandQueue);

	// Stop Timer
	sdkStopTimer(&timer);
	timeOnGPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);

	// Read result back from device to host
	const size_t origin[3] = { 0, 0, 0 };
	const size_t region[3] = { (size_t)newImage->width, (size_t)newImage->height, 1 };

	result = clEnqueueReadImage(oclCommandQueue, d_highlightedChanges, CL_TRUE, origin, region, 0, 0, h_highlightedChangePixArr, 0, NULL, NULL);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clEnqueueReadImage() Failed " << getErrorString(result) << "... Exiting !!!" << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}

	cout << endl << "Time Taken on GPU : " << timeOnGPU << " ms" << endl;

	convertPixelArrToBitmap(detectedChanges, h_highlightedChangePixArr, size);

	cleanup();
}

string getErrorString(cl_int error)
{
	switch(error)
	{
		// Run-time and JIT Errors
		case 0: return "CL_SUCCESS";
		case -1: return "CL_DEVICE_NOT_FOUND";
		case -2: return "CL_DEVICE_NOT_AVAILABLE";
		case -3: return "CL_COMPILER_NOT_AVAILABLE";
		case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case -5: return "CL_OUT_OF_RESOURCES";
		case -6: return "CL_OUT_OF_HOST_MEMORY";
		case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case -8: return "CL_MEM_COPY_OVERLAP";
		case -9: return "CL_IMAGE_FORMAT_MISMATCH";
		case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case -11: return "CL_BUILD_PROGRAM_FAILURE";
		case -12: return "CL_MAP_FAILURE";
		case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case -15: return "CL_COMPILE_PROGRAM_FAILURE";
		case -16: return "CL_LINKER_NOT_AVAILABLE";
		case -17: return "CL_LINK_PROGRAM_FAILURE";
		case -18: return "CL_DEVICE_PARTITION_FAILED";
		case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		// Compile-time errors
		case -30: return "CL_INVALID_VALUE";
		case -31: return "CL_INVALID_DEVICE_TYPE";
		case -32: return "CL_INVALID_PLATFORM";
		case -33: return "CL_INVALID_DEVICE";
		case -34: return "CL_INVALID_CONTEXT";
		case -35: return "CL_INVALID_QUEUE_PROPERTIES";
		case -36: return "CL_INVALID_COMMAND_QUEUE";
		case -37: return "CL_INVALID_HOST_PTR";
		case -38: return "CL_INVALID_MEM_OBJECT";
		case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40: return "CL_INVALID_IMAGE_SIZE";
		case -41: return "CL_INVALID_SAMPLER";
		case -42: return "CL_INVALID_BINARY";
		case -43: return "CL_INVALID_BUILD_OPTIONS";
		case -44: return "CL_INVALID_PROGRAM";
		case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
		case -46: return "CL_INVALID_KERNEL_NAME";
		case -47: return "CL_INVALID_KERNEL_DEFINITION";
		case -48: return "CL_INVALID_KERNEL";
		case -49: return "CL_INVALID_ARG_INDEX";
		case -50: return "CL_INVALID_ARG_VALUE";
		case -51: return "CL_INVALID_ARG_SIZE";
		case -52: return "CL_INVALID_KERNEL_ARGS";
		case -53: return "CL_INVALID_WORK_DIMENSION";
		case -54: return "CL_INVALID_WORK_GROUP_SIZE";
		case -55: return "CL_INVALID_WORK_ITEM_SIZE";
		case -56: return "CL_INVALID_GLOBAL_OFFSET";
		case -57: return "CL_INVALID_EVENT_WAIT_LIST";
		case -58: return "CL_INVALID_EVENT";
		case -59: return "CL_INVALID_OPERATION";
		case -60: return "CL_INVALID_GL_OBJECT";
		case -61: return "CL_INVALID_BUFFER_SIZE";
		case -62: return "CL_INVALID_MIP_LEVEL";
		case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
		case -64: return "CL_INVALID_PROPERTY";
		case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
		case -66: return "CL_INVALID_COMPILER_OPTIONS";
		case -67: return "CL_INVALID_LINKER_OPTIONS";
		case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

		// Extension Errors
		case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
		case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
		case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
		case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
		case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
		case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";

		default: 
		return "Unknown OpenCL error";
    }
}

void cleanup(void)
{
	if (d_highlightedChanges)
	{
		clReleaseMemObject(d_highlightedChanges);
		d_highlightedChanges = NULL;
	}

	if (d_newImage)
	{
		clReleaseMemObject(d_newImage);
		d_newImage = NULL;
	}

	if (d_oldImage)
	{
		clReleaseMemObject(d_oldImage);
		d_oldImage = NULL;
	}

	if (oclKernel)
	{
		clReleaseKernel(oclKernel);
		oclKernel = NULL;
	}
	
	if (oclProgram)
	{
		clReleaseProgram(oclProgram);
		oclProgram = NULL;
	}

	if (oclCommandQueue)
	{
		clReleaseCommandQueue(oclCommandQueue);
		oclCommandQueue = NULL;
	}

	if (oclContext)
	{
		clReleaseContext(oclContext);
		oclContext = NULL;
	}

	if (oclDeviceIds)
	{
		free(oclDeviceIds);
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

	if (logFile)
	{
		logFile.close();
	}
}
