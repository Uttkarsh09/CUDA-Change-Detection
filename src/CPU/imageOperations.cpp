#include <iostream>
#include "FreeImage.h"
#include <assert.h>
#include "common/dataTypes.hpp"

#define DIFFERENCE_THRESHOLD 60

using namespace std;


// * lpsz -> Long pointer to Null Terminated String
// * dib -> device independent bitmap
FIBITMAP* ImageFormatIndependentLoader(const char* lpszPathName, int flag){
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;

	// ? The second argument is not used by FreeImgae!!
	fif = FreeImage_GetFileType(lpszPathName, 0);

	// ? this means there is no signature, try to guess it from the file name (.png, ...)
	if(fif == FIF_UNKNOWN)
		fif = FreeImage_GetFIFFromFilename(lpszPathName);

	if((fif != FIF_UNKNOWN) && FreeImage_FIFSupportsReading(fif)){
		FIBITMAP *dib = FreeImage_Load(fif, lpszPathName, flag);
		return dib;
	}
	
	// ? fif == FIF_UNKNOWN, so we terminate
	cout << "ERROR -> FILE IMAGE FORMAT UNKNOWN";
	exit(1);
}


string mapIDToImageFormatName(FREE_IMAGE_FORMAT id){
	string fif = FreeImage_GetFormatFromFIF(id);
	return (fif == "") ? "!!! FIF_UNKNOWN !!!" : fif;
}


string mapIDToColorTypeName(FREE_IMAGE_COLOR_TYPE id){
	switch(id){
		case 0: return "FIC_MINISWHITE";
		case 1: return "FIC_MINISBLACK";
		case 2: return "FIC_RGB       ";
		case 3: return "FIC_PALETTE   ";
		case 4: return "FIC_RGBALPHA  ";
		case 5: return "FIC_CMYK      ";
		default: return "FIF_UNKNOWN";
	}
}


void printImageData(IMAGE_DATA img){
	cout << "Address \t= " << img.address << endl;
	cout << "Resolution \t= " << img.width << "x" << img.height << endl;
	cout << "Bits Per Pixel \t= " << img.bpp << endl;
	cout << "Memory Size \t= " << img.memorySize << endl;
	cout << "Color Type \t= " << img.colorType << " -> " << mapIDToColorTypeName(img.colorType) << endl;
	cout << "Image Format \t= " << img.imageFormat << " -> " << mapIDToImageFormatName(img.imageFormat) << endl;
	cout << endl;
}


void saveImage(IMAGE_DATA imgData, string address=""){
	if(address == "")	// ? else assuing address is given to imgData
		address = imgData.address;

	bool saved = FreeImage_Save(imgData.imageFormat, imgData.dib, address.c_str(), 0);

	if(!saved){
		cout << endl << "~~~~~~~~~~" << endl;
		perror("Can't save the file");
		cout << endl << "~~~~~~~~~~" << endl;
		exit(1);
	} 
	
	cout << "Image Saved Successfully at " << imgData.address << endl;
}


void populateImageData(IMAGE_DATA *imgData){
	imgData->width = FreeImage_GetWidth(imgData->dib);
	imgData->height = FreeImage_GetHeight(imgData->dib);
 	imgData->bpp = FreeImage_GetBPP(imgData->dib);
	imgData->memorySize = FreeImage_GetMemorySize(imgData->dib);	
	imgData->colorType = FreeImage_GetColorType(imgData->dib); 

	imgData->imageFormat = FreeImage_GetFileType(imgData->address.c_str(), 0);

	if(imgData->imageFormat == FIF_UNKNOWN){
		imgData->imageFormat = FreeImage_GetFIFFromFilename(imgData->address.c_str());
	
		if(imgData->imageFormat == FIF_UNKNOWN){
			cout << "Can't get FIF (Free Image Format)";
			exit(1);
		}
	}
}


// ! THIS ONLY SUPPORTS DETECTION FOR FIC_MINISBLACK TYPE IMAGES - Doing this deliberately
FIBITMAP* detectChanges(IMAGE_DATA img1, IMAGE_DATA img2){
	if((img1.colorType != FIC_MINISBLACK) || (img2.colorType != FIC_MINISBLACK)){
		cout << "ERROR: Color type of the images is not FIC_MINISBLACK" << endl;
		exit(1);
	}

	if(img1.bpp != img2.bpp){
		cout << "ERROR: Bits Per Pixel are different for the images can't compare the changes";
		exit(1);
	}

	if((img1.height != img2.height) || (img1.width != img2.width)){
		cout << "ERROR: Resolution of the images are not equal, pass images with same resolution" << endl;
		cout << "Img1 -> " << img1.width << " x " << img1.height << endl;
		cout << "Img2 -> " << img2.width << " x " << img2.height << endl;
		exit(1);
	}

	FIBITMAP *differenceBitmap = FreeImage_Allocate(img1.width, img1.height, img1.bpp);
	BYTE img1Pixel, img2Pixel, difference;


	for(int y=0 ; y<img1.height ; y++){
		for(int x=0 ; x<img1.width; x++){
			assert(FreeImage_GetPixelIndex(img1.dib, x, y, &img1Pixel));
			assert(FreeImage_GetPixelIndex(img2.dib, x, y, &img2Pixel));
			difference = abs(img1Pixel - img2Pixel);

			if((int)difference != 0){
				// cout << (int)difference << "\t";
			}
			FreeImage_SetPixelIndex(differenceBitmap, x, y, &difference);
		}
	}

	return differenceBitmap;
}


void copyImage(IMAGE_DATA *targetImage, IMAGE_DATA *sourceImage, string targetImageAddress=""){
	if(targetImageAddress == ""){
		// ? below line is to add "_copy" after the image name
		targetImage->address = sourceImage->address.substr(0, sourceImage->address.find_last_of(".")) + "_copy" + sourceImage->address.substr(sourceImage->address.find_last_of("."));
	}
	else {
		targetImage->address = targetImageAddress;
	}

	targetImage->bpp = sourceImage->bpp;
	targetImage->colorType = sourceImage->colorType;
	targetImage->dib = sourceImage->dib;
	targetImage->height = sourceImage->height;
	targetImage->width = sourceImage->width;
	targetImage->imageFormat = sourceImage->imageFormat;
}


void highlightChangesInImage(IMAGE_DATA *img, FIBITMAP *differenceBitmap){
	assert(FreeImage_GetBPP(differenceBitmap) == 8);
	assert(img->colorType == FIC_RGB);
	assert(img->height == FreeImage_GetHeight(differenceBitmap));
	assert(img->width == FreeImage_GetWidth(differenceBitmap));

	RGBQUAD color;

	for(int y=0 ; y<img->height ; y++){
		for(int x=0 ; x<img->width; x++){
			FreeImage_GetPixelIndex(differenceBitmap, x, y, &color.rgbRed);
			if(color.rgbRed >= DIFFERENCE_THRESHOLD){
				color.rgbRed = 255;
				color.rgbBlue = 0;
				color.rgbGreen = 0;
				FreeImage_SetPixelColor(img->dib, x, y, &color);
			}
		}
	}
}


void convertToRGBGreyscale(IMAGE_DATA *img, IMAGE_DATA *greyscaleImg){
	assert(greyscaleImg->bpp == 8);
	assert((greyscaleImg->colorType == FIC_MINISBLACK) || (greyscaleImg->colorType == FIC_MINISWHITE));
	BYTE bits;
	RGBQUAD color;
	
	for(int y=0 ; y<img->height ; y++){
		for(int x=0 ; x<img->width ; x++){
			FreeImage_GetPixelIndex(greyscaleImg->dib, x, y, &bits);
			color.rgbRed = bits;
			color.rgbGreen = bits;
			color.rgbBlue = bits;
			FreeImage_SetPixelColor(img->dib, x, y, &color);
		}
	}
}
