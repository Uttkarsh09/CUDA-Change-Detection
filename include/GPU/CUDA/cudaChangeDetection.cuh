#pragma once

#include "../../common/systemMacros.hpp"

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
void printCUDADeviceProperties(void);
void runOnGPU(void);
void cleanup(void);
