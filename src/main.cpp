// * dib -> Device Independent Bitmap - bitmap datatyp of FreeImage
// * bitmap -> normal BYTE bitmap (array of BYTE)

#include "../include/headers.hpp"

int main()
{
	IMAGE_DATA oldImage, newImage;
	BYTE *oldImageBitmap, *newImageBitmap, *highlightChangesBitmap;

	oldImage.address = getImagePath("old.png");
	newImage.address = getImagePath("new.png");

	oldImage.dib = imageFormatIndependentLoader(oldImage.address.c_str(), 0);
	newImage.dib = imageFormatIndependentLoader(newImage.address.c_str(), 0);

	populateImageData(&oldImage);
	populateImageData(&newImage);

	if((oldImage.height != newImage.height) || (oldImage.width != newImage.width))
	{
		cout << "Error: Image Dimentions not same";
		exit(1);
	}
	
	oldImage.bitmap = (BYTE*)malloc(oldImage.height * oldImage.bitmapWidth);
	newImage.bitmap = (BYTE*)malloc(newImage.height * newImage.bitmapWidth);
	
	FreeImage_ConvertToRawBits(
		oldImage.bitmap, 
		oldImage.dib, 
		oldImage.bitmapWidth,
		oldImage.bpp, 
		FI_RGBA_RED_MASK, 
		FI_RGBA_GREEN_MASK,
		FI_RGBA_BLUE_MASK,
		TRUE
	);

	FreeImage_ConvertToRawBits(
		newImage.bitmap,
		newImage.dib,
		newImage.bitmapWidth,
		newImage.bpp,
		FI_RGBA_RED_MASK,
		FI_RGBA_GREEN_MASK,
		FI_RGBA_BLUE_MASK,
		TRUE
	);

	runOnCPU(&oldImage, &newImage, DIFFERENCE_THRESHOLD);

	// getPlatformInfo();
	
	return 0;
}
