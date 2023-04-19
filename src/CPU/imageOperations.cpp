#include "../../include/CPU/imageOperations.hpp"


void CPUChangeDetection(Pixel *oldImagePixelArr, Pixel *newImagePixelArr, Pixel *highlightedChangePixelArr, uint8_t threshold, int width, int height)
{
	uint8_t oldGreyVal, newGreyVal, difference;

	for(int j=0 ; j<height ; j++)
	{
		for(int i=0 ; i<width; i++)
		{
			oldGreyVal = (uint8_t) (
				(0.3 * (uint8_t)oldImagePixelArr[(j * width) + i].red) + 
				(0.59 * (uint8_t)oldImagePixelArr[(j * width) + i].green) + 
				(0.11 * (uint8_t)oldImagePixelArr[(j * width) + i].blue)
			);

			newGreyVal = (uint8_t) (
				(0.3 * (uint8_t)newImagePixelArr[(j * width) + i].red) + 
				(0.59 * (uint8_t)newImagePixelArr[(j * width) + i].green) + 
				(0.11 * (uint8_t)newImagePixelArr[(j * width) + i].blue)
			);

			difference = abs(oldGreyVal - newGreyVal);
			
			if(difference >= threshold)
			{
				highlightedChangePixelArr[(j * width) + i].red = 255;
				highlightedChangePixelArr[(j * width) + i].green = 0;
				highlightedChangePixelArr[(j * width) + i].blue = 0;
			}
			else 
			{
				highlightedChangePixelArr[(j * width) + i].red = oldGreyVal;
				highlightedChangePixelArr[(j * width) + i].green = oldGreyVal;
				highlightedChangePixelArr[(j * width) + i].blue = oldGreyVal;
			}
		}
	}
}