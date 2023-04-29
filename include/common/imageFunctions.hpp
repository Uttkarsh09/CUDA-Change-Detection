#pragma once

#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
using namespace std;

#include "devicePlatform.hpp"
#include "dataTypes.hpp"
#include "systemMacros.hpp"
#include "FreeImage.h"

#if (PLATFORM == 1)
    #pragma comment(lib, "FreeImage.lib")
#endif

void FreeImageErrorHandler(FREE_IMAGE_FORMAT fif, const char *message);

void populateImageData(ImageData *imageData);

void printImageData(ImageData *image);

FIBITMAP* imageFormatIndependentLoader(const char* lpszPathName, int flag);

void saveImage(FIBITMAP *dib, FREE_IMAGE_FORMAT imageFormat, string address);

void convertBitmapToPixelArr(Pixel *pixelArr, uint8_t *bitmap, size_t size);

void convertPixelArrToBitmap(uint8_t *bitmap, Pixel *pixelArr, size_t size);

string getOSPath(vector<string>);