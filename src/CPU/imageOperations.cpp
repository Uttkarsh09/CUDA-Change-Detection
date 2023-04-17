#include "../../include/CPU/imageOperations.hpp"


string mapIDToImageFormatName(FREE_IMAGE_FORMAT id)
{
	string fif = FreeImage_GetFormatFromFIF(id);
	return (fif == "") ? "!!! FIF_UNKNOWN !!!" : fif;
}


string mapIDToColorTypeName(FREE_IMAGE_COLOR_TYPE id)
{
	switch(id)
	{
		case 0: return "FIC_MINISWHITE";
		case 1: return "FIC_MINISBLACK";
		case 2: return "FIC_RGB       ";
		case 3: return "FIC_PALETTE   ";
		case 4: return "FIC_RGBALPHA  ";
		case 5: return "FIC_CMYK      ";
		default: return "FIF_UNKNOWN";
	}
}


void printImageData(IMAGE_DATA img)
{
	cout << "Address \t= " << img.address << endl;
	cout << "Resolution \t= " << img.width << "x" << img.height << endl;
	cout << "Bits Per Pixel \t= " << img.bpp << endl;
	cout << "Memory Size \t= " << img.memorySize << endl;
	cout << "Color Type \t= " << img.colorType << " -> " << mapIDToColorTypeName(img.colorType) << endl;
	cout << "Image Format \t= " << img.imageFormat << " -> " << mapIDToImageFormatName(img.imageFormat) << endl;
	cout << endl;
}


void CPUChangeDetection(BYTE *oldImageBitmap, BYTE *newImageBitmap, BYTE *highlightChangesBitmap, int bitmapWidth, int width, int height, int threshold)
{
	int oldGreyVal, newGreyVal, difference;
	BYTE *oldImagePixels, *newImagePixels, *highlightChangePixels;

	for(int j=0 ; j<height ; j++)
	{
		oldImagePixels = (BYTE*)oldImageBitmap;
		newImagePixels = (BYTE*)newImageBitmap;
		highlightChangePixels = (BYTE*)highlightChangesBitmap;

		for(int i=0 ; i<width; i++)
		{
			oldGreyVal = (int) (
				(0.3 * (int)oldImagePixels[FI_RGBA_RED]) + 
				(0.59 * (int)oldImagePixels[FI_RGBA_GREEN]) + 
				(0.11 * (int) oldImagePixels[FI_RGBA_BLUE])
			);

			newGreyVal = (int) (
				(0.3 * (int)newImagePixels[FI_RGBA_RED]) + 
				(0.59 * (int)newImagePixels[FI_RGBA_GREEN]) + 
				(0.11 * (int) newImagePixels[FI_RGBA_BLUE])
			);

			difference = abs(oldGreyVal - newGreyVal);
			
			if(difference >= threshold)
			{
				highlightChangePixels[FI_RGBA_RED] = 255;
				highlightChangePixels[FI_RGBA_GREEN] = 0;
				highlightChangePixels[FI_RGBA_BLUE] = 0;
			} 
			else 
			{
				highlightChangePixels[FI_RGBA_RED] = oldGreyVal;
				highlightChangePixels[FI_RGBA_GREEN] = oldGreyVal;
				highlightChangePixels[FI_RGBA_BLUE] = oldGreyVal;
			}
			

			oldImagePixels += 3;
			newImagePixels += 3;
			highlightChangePixels += 3;
		}

		oldImageBitmap += bitmapWidth;
		newImageBitmap += bitmapWidth;
		highlightChangesBitmap += bitmapWidth;
	}
}