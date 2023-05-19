// * dib -> Device Independent Bitmap - bitmap datatyp of FreeImage
// * bitmap -> normal uint8_t bitmap (array of uint8_t)

#include "../include/headers.hpp"

int main(int argc, char* argv[])
{
	ImageData oldImage, newImage;
	string imageResolution = argv[1];
	string printDevProp = argv[2];

	oldImage.address = getOSPath({"images", imageResolution, imageResolution+"_old.png"});
	newImage.address = getOSPath({"images", imageResolution, imageResolution+"_new.png"});

	oldImage.dib = imageFormatIndependentLoader(oldImage.address.c_str(), 0);
	newImage.dib = imageFormatIndependentLoader(newImage.address.c_str(), 0);

	populateImageData(&oldImage);
	populateImageData(&newImage);

	// printImageData(&oldImage);
	// printImageData(&newImage);

	if((oldImage.height != newImage.height) || (oldImage.width != newImage.width))
	{
		cerr << "Error: Image Dimentions not same";
		exit(EXIT_FAILURE);
	}
	
	oldImage.bitmap = (uint8_t*)malloc(oldImage.height * oldImage.pitch);
	newImage.bitmap = (uint8_t*)malloc(newImage.height * newImage.pitch);
	
	FreeImage_ConvertToRawBits(
		oldImage.bitmap, 
		oldImage.dib, 
		oldImage.pitch,
		oldImage.bpp, 
		FI_RGBA_RED_MASK, 
		FI_RGBA_GREEN_MASK,
		FI_RGBA_BLUE_MASK,
		TRUE
	);

	FreeImage_ConvertToRawBits(
		newImage.bitmap,
		newImage.dib,
		newImage.pitch,
		newImage.bpp,
		FI_RGBA_RED_MASK,
		FI_RGBA_GREEN_MASK,
		FI_RGBA_BLUE_MASK,
		TRUE
	);

	uint8_t *GPU_DetectedChangesBitmap, *CPU_DetectedChangesBitmap;
	FIBITMAP *GPU_DetectedChangesDib, *CPU_DetectedChangesDib;
	string CPU_ImageAddress, GPU_ImageAddress;
	
	CPU_ImageAddress = getOSPath({"images", imageResolution, imageResolution+"_CPU_Highlighted_Changes.png"});
	GPU_ImageAddress = getOSPath({"images", imageResolution, imageResolution+"_GPU_Highlighted_Changes.png"});

	CPU_DetectedChangesBitmap = (uint8_t*)malloc(oldImage.height * oldImage.pitch);
	GPU_DetectedChangesBitmap = (uint8_t*)malloc(oldImage.height * oldImage.pitch);

	if (printDevProp == "1")
	{
		printDeviceProperties();
	}
	
	cout << endl << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ " + imageResolution + "x" + imageResolution + " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
	runOnCPU(&oldImage, &newImage, DIFFERENCE_THRESHOLD, CPU_DetectedChangesBitmap);
	runOnGPU(&oldImage, &newImage, DIFFERENCE_THRESHOLD, GPU_DetectedChangesBitmap);
	

	CPU_DetectedChangesDib = FreeImage_ConvertFromRawBits(
		CPU_DetectedChangesBitmap, 
		oldImage.width, 
		oldImage.height, 
		oldImage.pitch, 
		oldImage.bpp, 
		FI_RGBA_RED_MASK, 
		FI_RGBA_GREEN_MASK, 
		FI_RGBA_BLUE_MASK, 
		TRUE
	);


	GPU_DetectedChangesDib = FreeImage_ConvertFromRawBits(
			GPU_DetectedChangesBitmap, 
			oldImage.width, 
			oldImage.height, 
			oldImage.pitch, 
			oldImage.bpp, 
			FI_RGBA_RED_MASK, 
			FI_RGBA_GREEN_MASK, 
			FI_RGBA_BLUE_MASK, 
			TRUE
		);

	
	saveImage(CPU_DetectedChangesDib, oldImage.imageFormat, CPU_ImageAddress);
	saveImage(GPU_DetectedChangesDib, oldImage.imageFormat, GPU_ImageAddress);
	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl << endl;
	
	free(GPU_DetectedChangesBitmap);
	free(CPU_DetectedChangesBitmap);
	
	return 0;
}
