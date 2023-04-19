#pragma once

#include "imageOperations.hpp"
#include "../common/dataTypes.hpp"
#include "../common/imageFunctions.hpp"
#include "../common/helper_timer.h"

void runOnCPU(ImageData *oldImage, ImageData *newImage, int threshold, uint8_t *detectedChanges);
