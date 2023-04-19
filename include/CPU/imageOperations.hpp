#pragma once

#include <iostream>
#include <assert.h>
#include "../common/FreeImage.h"
#include "../common/dataTypes.hpp"
using namespace std;

#define DIFFERENCE_THRESHOLD 60

void CPUChangeDetection(uint8_t *oldImageBitmap, uint8_t *newImageBitmap, uint8_t *highlightChangesBitmap, int pitch, int width, int height, int threshold);