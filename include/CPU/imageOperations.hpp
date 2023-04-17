#pragma once

#include <iostream>
#include <assert.h>
#include "../common/FreeImage.h"
#include "../common/dataTypes.hpp"
using namespace std;

#define DIFFERENCE_THRESHOLD 60

void printImageData(IMAGE_DATA img);

void CPUChangeDetection(BYTE *oldImageBitmap, BYTE *newImageBitmap, BYTE *highlightChangesBitmap, int bitmapWidth, int width, int height, int threshold);