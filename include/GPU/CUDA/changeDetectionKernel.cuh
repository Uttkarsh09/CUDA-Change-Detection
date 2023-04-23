#pragma once

#include <cuda.h>
#include "../../common/dataTypes.hpp"

#if (PLATFORM == 1)
    #pragma comment(lib, "cuda.lib")
    #pragma comment(lib, "cudart.lib")
#endif


__global__ void detectChanges(Pixel *oldImage, Pixel *newImage, Pixel *highlightedChanges, uint8_t threshold, int count);
