#pragma once

#include "../../common/devicePlatform.hpp"

#if (PLATFORM == 1)
    #include <cuda.h>
    #pragma comment(lib, "cuda.lib")
    #pragma comment(lib, "cudart.lib")
#else
    #include <cuda.h>
#endif

#include <iostream>
#include <math.h>
using namespace std;

// Function Protoypes
void getPlatformInfo(void);
void printCUDADeviceProperties(void);
