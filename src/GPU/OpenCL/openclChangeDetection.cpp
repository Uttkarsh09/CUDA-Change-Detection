#include "../../../include/GPU/OpenCL/openclChangeDetection.hpp"
#include "../../../include/common/helper_timer.h"

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

cl_image_desc oclImageDescriptor_oldImage;
cl_image_desc oclImageDescriptor_newImage;
cl_image_desc oclImageDescriptor_highlightedChanges;
cl_image_format oclImageFormat;

int gpuChoice = -1;

Pixel *h_oldImagePixArr, *h_newImagePixArr, *h_highlightedChangePixArr;
// Pixel *d_oldImagePixArr, *d_newImagePixArr, *d_highlightedChangePixArr;
cl_mem d_oldImage, d_newImage, d_highlightedChanges;

ofstream logFile;

const char* oclSourceCode =
"__kernel void oclChangeDetection(__read_only image2d_t oldImage, __read_only image2d_t newImage, __write_only image2d_t highlightedChanges, unsigned char threshold, int count)" \
"{ " \
    "long pixelId = get_global_id(0);" \
    "unsigned char oldGreyVal, newGreyVal, difference;" \

	"if (pixelId < count)" \
	"{" \
		"oldGreyVal = (unsigned char)((0.3 * (unsigned char)oldImagePixelArr[pixelId].red) + (0.59 * (unsigned char)oldImagePixelArr[pixelId].green) + (0.11 * (unsigned char) oldImagePixelArr[pixelId].blue));" \

		"newGreyVal = (unsigned char)((0.3 * (unsigned char)newImagePixelArr[pixelId].red) + (0.59 * (unsigned char)newImagePixelArr[pixelId].green) + (0.11 * (unsigned char)newImagePixelArr[pixelId].blue));" \

		"difference = abs(oldGreyVal - newGreyVal);" \

		"if (difference >= threshold)" \
		"{" \
			"highlightedChangePixelArr[pixelId].red = 255;" \
			"highlightedChangePixelArr[pixelId].green = 0;" \
			"highlightedChangePixelArr[pixelId].blue = 0;" \
		"}" \
		"else" \
		"{" 
			"highlightedChangePixelArr[pixelId].red = oldGreyVal;" \
			"highlightedChangePixelArr[pixelId].green = oldGreyVal;" \
			"highlightedChangePixelArr[pixelId].blue = oldGreyVal;" \
		"}" \
	"}"\

"}";

void getOpenCLPlatforms(void)
{
	// Code
    result = clGetPlatformIDs(1, &oclPlatformId, NULL);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clGetPlatformIDs() Failed : " << result << endl;
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
		cerr << endl << "clGetDeviceIDs() Failed : " << result << endl;
		cleanup();
        exit(EXIT_FAILURE);
	}
	else if (devCount == 0)
	{
		cerr << endl << "No OpenCL Supported Device Found On This System !!!" << endl;
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
	// Code
    oclContext = clCreateContext(NULL, 1, &oclDeviceId, NULL, NULL, &result);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clCreateContext() Failed : " << result << endl;
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
		cerr << endl << "clCreateCommandQueue() Failed : " << result << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}
}

void createOpenCLImageStructure(ImageData* oldImage, ImageData* newImage)
{
	// Code

	// Initialize cl_image_desc structure for old image
	oclImageDescriptor_oldImage.image_type = CL_MEM_OBJECT_IMAGE2D;
	oclImageDescriptor_oldImage.image_width = oldImage->width;
	oclImageDescriptor_oldImage.image_height = oldImage->height;
	oclImageDescriptor_oldImage.image_depth = 0;
	oclImageDescriptor_oldImage.image_array_size = 0;
	oclImageDescriptor_oldImage.image_row_pitch = 0;
	oclImageDescriptor_oldImage.image_slice_pitch = 0;
	oclImageDescriptor_oldImage.num_mip_levels = 0;
	oclImageDescriptor_oldImage.num_samples = 0;
	oclImageDescriptor_oldImage.buffer = NULL;

	// Initialize cl_image_desc structure for new image
	oclImageDescriptor_newImage.image_type = CL_MEM_OBJECT_IMAGE2D;
	oclImageDescriptor_newImage.image_width = newImage->width;
	oclImageDescriptor_newImage.image_height = newImage->height;
	oclImageDescriptor_newImage.image_depth = 0;
	oclImageDescriptor_newImage.image_array_size = 0;
	oclImageDescriptor_newImage.image_row_pitch = 0;
	oclImageDescriptor_newImage.image_slice_pitch = 0;
	oclImageDescriptor_newImage.num_mip_levels = 0;
	oclImageDescriptor_newImage.num_samples = 0;
	oclImageDescriptor_newImage.buffer = NULL;

	// Initialize cl_image_desc structure for highlighted changes
	oclImageDescriptor_highlightedChanges.image_type = CL_MEM_OBJECT_IMAGE2D;
	oclImageDescriptor_highlightedChanges.image_width = newImage->width;
	oclImageDescriptor_highlightedChanges.image_height = newImage->height;
	oclImageDescriptor_highlightedChanges.image_depth = 0;
	oclImageDescriptor_highlightedChanges.image_array_size = 0;
	oclImageDescriptor_highlightedChanges.image_row_pitch = 0;
	oclImageDescriptor_highlightedChanges.image_slice_pitch = 0;
	oclImageDescriptor_highlightedChanges.num_mip_levels = 0;
	oclImageDescriptor_highlightedChanges.num_samples = 0;
	oclImageDescriptor_highlightedChanges.buffer = NULL;

	// Initialize cl_image_format structure
	oclImageFormat.image_channel_order = CL_RGBA;
	oclImageFormat.image_channel_data_type = CL_UNSIGNED_INT8;

	// Create Old Input Image
	d_oldImage = clCreateImage(oclContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &oclImageFormat, &oclImageDescriptor_oldImage, (void*)h_oldImagePixArr, &result);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clCreateImage() Failed For Old Input Image ... Exiting !!!" << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}

	// Create New Input Image
	d_newImage = clCreateImage(oclContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &oclImageFormat, &oclImageDescriptor_newImage, (void*)h_newImagePixArr, &result);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clCreateImage() Failed For New Input Image ... Exiting !!!" << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}

	// Create Ouput Image (Highlighted Changes)
	d_highlightedChanges = clCreateImage(oclContext, CL_MEM_WRITE_ONLY, &oclImageFormat, &oclImageDescriptor_highlightedChanges, NULL, &result);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clCreateImage() Failed For Ouput Image ... Exiting !!!" << endl;
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
		cerr << endl << "clCreateProgramWithSource() Failed : " << result << endl;
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
			cerr << "Failed to create log !!!" << endl;
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

void createOpenCLKernel(ImageData* oldImage, ImageData* newImage, size_t size, unsigned char threshold, int count)
{
	oclKernel = clCreateKernel(oclProgram, "oclChangeDetection", &result);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clCreateKernel() Failed : " << result << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), d_oldImage);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clSetKernelArg() Failed For 1st Argument : " << result << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = clSetKernelArg(oclKernel, 1, sizeof(cl_mem), d_newImage);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clSetKernelArg() Failed For 2nd Argument : " << result << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = clSetKernelArg(oclKernel, 2, sizeof(cl_mem), d_highlightedChanges);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clSetKernelArg() Failed For 3rd Argument : " << result << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = clSetKernelArg(oclKernel, 3, sizeof(cl_uchar), (void*)&threshold);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clSetKernelArg() Failed For 4th Argument : " << result << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = clSetKernelArg(oclKernel, 4, sizeof(cl_int), (void*)&count);
	if (result != CL_SUCCESS)
	{
		cerr << endl << "clSetKernelArg() Failed For 5th Argument : " << result << endl;
		cleanup();
		exit(EXIT_FAILURE);
	}
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

	// d_oldImagePixArr = (Pixel*)malloc(size * sizeof(Pixel));
	// if (d_oldImagePixArr == NULL)
	// {
	// 	cout << endl << "Failed to allocate memory for d_oldImagePixArr ... Exiting !!!" << endl;
	// 	cleanup();
	// 	exit(EXIT_FAILURE);
	// }

	// d_newImagePixArr = (Pixel*)malloc(size * sizeof(Pixel));
	// if (h_highlightedChangePixArr == NULL)
	// {
	// 	cout << endl << "Failed to allocate memory for d_newImagePixArr ... Exiting !!!" << endl;
	// 	cleanup();
	// 	exit(EXIT_FAILURE);
	// }

	// d_highlightedChangePixArr = (Pixel*)malloc(size * sizeof(Pixel));
	// if (h_highlightedChangePixArr == NULL)
	// {
	// 	cout << endl << "Failed to allocate memory for d_highlightedChangePixArr ... Exiting !!!" << endl;
	// 	cleanup();
	// 	exit(EXIT_FAILURE);
	// }

	createOpenCLContext();

	createOpenCLCommandQueue();

	createOpenCLImageStructure(oldImage, newImage);

	// createOpenCLProgram("oclChangeDetection.cl");
	
	// createOpenCLKernel(size * sizeof(Pixel), (unsigned char)threshold, size);

	//createOpenCLEnqueueWriteBuffer(size * sizeof(Pixel));

	cout << endl << "createOpenCLImageStructure() Done" << endl;

	// // Kernel Configuration
	// size_t global_size = 1024;

	// // Start Timer
	// StopWatchInterface* timer = NULL;
	// sdkCreateTimer(&timer);
	// sdkStartTimer(&timer);

	// result = clEnqueueNDRangeKernel(oclCommandQueue, oclKernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
	// if (result != CL_SUCCESS)
	// {
	// 	cout << endl << "clEnqueueNDRangeKernel() Failed : " << result << endl;
	// 	cleanup();
	// 	exit(EXIT_FAILURE);
	// }

	// clFinish(oclCommandQueue);

	// // Stop Timer
	// sdkStopTimer(&timer);
	// timeOnGPU = sdkGetTimerValue(&timer);
	// sdkDeleteTimer(&timer);

	// // Read result back from device to host
	// result = clEnqueueReadBuffer(oclCommandQueue, deviceOutput, CL_TRUE, 0, size, h_highlightedChangePixArr, 0, NULL, NULL);
	// if (result != CL_SUCCESS)
	// {
	// 	cout << endl << "clEnqueueReadBuffer() Failed : " << result << endl;
	// 	cleanup();
	// 	exit(EXIT_FAILURE);
	// }

	// cout << endl << "Time Taken on GPU : " << timeOnGPU << " ms" << endl;

	cleanup();
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
