#include "../../include/CPU/imageOperations.hpp"


void CPUChangeDetection(uint8_t *oldImageBitmap, uint8_t *newImageBitmap, uint8_t *highlightChangesBitmap, int pitch, int width, int height, int threshold)
{
	int oldGreyVal, newGreyVal, difference;
	uint8_t *oldImagePixels, *newImagePixels, *highlightChangePixels;

	for(int j=0 ; j<height ; j++)
	{
		oldImagePixels = (uint8_t*)oldImageBitmap;
		newImagePixels = (uint8_t*)newImageBitmap;
		highlightChangePixels = (uint8_t*)highlightChangesBitmap;

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

		oldImageBitmap += pitch;
		newImageBitmap += pitch;
		highlightChangesBitmap += pitch;
	}
}