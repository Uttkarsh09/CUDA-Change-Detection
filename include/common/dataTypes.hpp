#pragma once

#include <iostream>
#include "FreeImage.h"
using namespace std;

typedef struct Image_Metadata 
{
	int width = -1;
	int height = -1;
	int bpp = -1; // ? Bits Per Pixel
	unsigned int memorySize = 0;					// ? -> values can be a standard approximation, may vary between using different C++ standard libs
	int bitmapWidth;
	BYTE *bitmap;
	string address = "unknown";
	FREE_IMAGE_FORMAT imageFormat = FIF_UNKNOWN;
	FIBITMAP *dib;
	FREE_IMAGE_COLOR_TYPE colorType; 				// ? One of -> FIC_MINISWHITE-0 | FIC_MINISBLACK-1 | FIC_RGB-2 | FIC_PALETTE-3 | FIC_RGBALPHA-4 | FIC_CMYK-5

} IMAGE_DATA;
