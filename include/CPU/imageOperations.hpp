#pragma once

#include <iostream>
#include <assert.h>
#include "../common/FreeImage.h"
#include "../common/dataTypes.hpp"
using namespace std;

#define DIFFERENCE_THRESHOLD 90

void CPUChangeDetection(Pixel *oldImagePixelArr, Pixel *newImagePixelArr, Pixel *highlightedChangePixelArr, uint8_t threshold, int width, int height);
