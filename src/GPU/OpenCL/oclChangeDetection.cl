__kernel void oclChangeDetection(Pixel *oldImagePixelArr, Pixel *newImagePixelArr, Pixel *highlightedChangePixelArr, uint8_t threshold, int count)
{
    long pixelID = get_global_id(0);
    uint8_t oldGreyVal, newGreyVal, difference;

	if (pixelId < count)
	{
		oldGreyVal = (uint8_t)(
			(0.3 * (uint8_t)oldImagePixelArr[pixelId].red) + 
			(0.59 * (uint8_t)oldImagePixelArr[pixelId].green) + 
			(0.11 * (uint8_t) oldImagePixelArr[pixelId].blue)
		);

		newGreyVal = (uint8_t)(
			(0.3 * (uint8_t)newImagePixelArr[pixelId].red) + 
			(0.59 * (uint8_t)newImagePixelArr[pixelId].green) + 
			(0.11 * (uint8_t)newImagePixelArr[pixelId].blue)
		);

		difference = abs(oldGreyVal - newGreyVal);

		if (difference >= threshold)
		{
			highlightedChangePixelArr[pixelId].red = 255;
			highlightedChangePixelArr[pixelId].green = 0;
			highlightedChangePixelArr[pixelId].blue = 0;
		}
		else 
		{
			highlightedChangePixelArr[pixelId].red = oldGreyVal;
			highlightedChangePixelArr[pixelId].green = oldGreyVal;
			highlightedChangePixelArr[pixelId].blue = oldGreyVal;
		}
	}
}
