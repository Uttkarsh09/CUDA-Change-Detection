__kernel void oclChangeDetection(__global uint8 *oldImagePixelArr, __global uint8 *newImagePixelArr, __global uint8 *highlightedChangePixelArr, uint8 threshold, int count) 
{  
    long pixelId = get_global_id(0); 
    uint8 oldGreyVal, newGreyVal, difference; 

	if (pixelId < count) 
	{ 
		oldGreyVal = (uint8)( 
			(0.3 * (uint8)oldImagePixelArr[pixelId].red) +
			(0.59 * (uint8)oldImagePixelArr[pixelId].green) +
			(0.11 * (uint8) oldImagePixelArr[pixelId].blue)
		);

		newGreyVal = (uint8)( 
			(0.3 * (uint8)newImagePixelArr[pixelId].red) + 
			(0.59 * (uint8)newImagePixelArr[pixelId].green) + 
			(0.11 * (uint8)newImagePixelArr[pixelId].blue) 
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
};
