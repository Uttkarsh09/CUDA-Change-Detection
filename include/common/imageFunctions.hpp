#pragma once

#include "systemMacros.hpp"

#if (PLATFORM == 1)
    #include <windows.h>
    #pragma comment(lib, "FreeImage.lib")
#endif

#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
using namespace std;

#include "devicePlatform.hpp"
#include "dataTypes.hpp"
#include "FreeImage.h"

// Function Declarations
string getOSPath(vector<string>);

void FreeImageErrorHandler(FREE_IMAGE_FORMAT fif, const char *message);

void populateImageData(ImageData *imageData);

void printImageData(ImageData *image);

FIBITMAP* imageFormatIndependentLoader(const char* lpszPathName, int flag);

void saveImage(FIBITMAP *dib, FREE_IMAGE_FORMAT imageFormat, string address);

void convertBitmapToPixelArr(Pixel *pixelArr, uint8_t *bitmap, size_t size);

void convertPixelArrToBitmap(uint8_t *bitmap, Pixel *pixelArr, size_t size);
