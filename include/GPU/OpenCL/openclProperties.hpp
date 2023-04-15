#pragma once

#include "../../common/devicePlatform.hpp"

#if (PLATFORM == 1) || (PLATFORM == 2)
    #include <CL/opencl.h>
    #pragma comment(lib, "OpenCL.lib")
#elif (PLATFORM == 2)
    #include <CL/opencl.h>
#else
    #include <OpenCL/opencl.h>
#endif

#include <iostream>
using namespace std;

// Function Declarations
void getPlatformInfo(void);
void printOpenCLDeviceProperties(void);
