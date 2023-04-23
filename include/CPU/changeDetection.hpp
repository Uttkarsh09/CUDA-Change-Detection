#pragma once

#include "../common/systemMacros.hpp"

#if (PLATFORM  == 1)
    #include <windows.h>
#endif

#include "../common/helper_timer.h"

#include "imageOperations.hpp"
#include "../common/dataTypes.hpp"
#include "../common/imageFunctions.hpp"


void runOnCPU(ImageData *oldImage, ImageData *newImage, int threshold, uint8_t *detectedChanges);
