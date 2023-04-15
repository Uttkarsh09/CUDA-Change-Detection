#pragma once

#include <iostream>
#include <assert.h>
#include "../common/FreeImage.h"
#include "../common/dataTypes.hpp"
using namespace std;

#define DIFFERENCE_THRESHOLD 60

string mapIDToColorTypeName(FREE_IMAGE_COLOR_TYPE);

FIBITMAP* ImageFormatIndependentLoader(const char* lpszPathName, int flag);

void printImageData(IMAGE_DATA img);

string mapIDToImageFormatName(FREE_IMAGE_FORMAT id);

string mapIDToColorTypeName(FREE_IMAGE_COLOR_TYPE id);

void populateImageData(IMAGE_DATA *imgData);

void saveImage(IMAGE_DATA imgData, string address="");

FIBITMAP* detectChanges(IMAGE_DATA img1, IMAGE_DATA img2);

void copyImage(IMAGE_DATA *targetImage, IMAGE_DATA *sourceImage, string targetImageAddress="");

void highlightChangesInImage(IMAGE_DATA *img, FIBITMAP *differenceBitmap);

void convertToRGBGreyscale(IMAGE_DATA *img, IMAGE_DATA *greyscaleImg);
