#include "../../include/CPU/changeDetection.hpp"

void runOnCPU(IMAGE_DATA *oldImage, IMAGE_DATA *newImage, int threshold)
{
	BYTE *highlightChangesBitmap, *startCpy;
	string highlightedImageAddress = getImagePath("CPU_Highlighted_Changes.png");
	FIBITMAP *highlightChangesDib;

	highlightChangesBitmap = (BYTE*)malloc(oldImage->height * oldImage->bitmapWidth);
	startCpy = highlightChangesBitmap;

	CPUChangeDetection(
		oldImage->bitmap,
		newImage->bitmap,
		highlightChangesBitmap,
		oldImage->bitmapWidth,
		oldImage->width,
		oldImage->height,
		threshold
	);

	highlightChangesDib = FreeImage_ConvertFromRawBits(
		startCpy, 
		oldImage->width, 
		oldImage->height, 
		oldImage->bitmapWidth, 
		oldImage->bpp, 
		FI_RGBA_RED_MASK, 
		FI_RGBA_GREEN_MASK, 
		FI_RGBA_BLUE_MASK, 
		TRUE
	);

	saveImage(highlightChangesDib, oldImage->imageFormat, highlightedImageAddress);

}