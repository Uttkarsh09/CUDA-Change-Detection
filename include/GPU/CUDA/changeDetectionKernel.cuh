#pragma once

#include <cuda.h>
#include "../../common/dataTypes.hpp"


__global__ void detectChanges(Pixel *oldImage, Pixel *newImage, Pixel *highlightedChanges, uint8_t threshold, int count);