#pragma once

#include "../../common/systemMacros.hpp"
#include "../../common/dataTypes.hpp"

#include <cuda.h>
#include <chrono>

#if (PLATFORM == 1)
    #pragma comment(lib, "cuda.lib")
    #pragma comment(lib, "cudart.lib")
#endif

#include <math.h>

// Function Protoypes
void printCUDADeviceProperties(void);
void runOnGPU(ImageData *oldImage, ImageData *newImage, int threshold, uint8_t *detectedChanges);
void cleanup(void);
