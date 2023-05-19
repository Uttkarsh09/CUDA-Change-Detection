#include "../../include/CPU/changeDetection.hpp"

Pixel *oldImagePixArr = NULL;
Pixel *newImagePixArr = NULL;
Pixel *highlightedChangePixArr = NULL;

void runOnCPU(ImageData *oldImage, ImageData *newImage, int threshold, uint8_t *detectedChanges)
{
	uint8_t *highlightChangesBitmap, *startCpy;
	FIBITMAP *highlightChangesDib;
	size_t size = (oldImage->height * oldImage->pitch)/3;
	float timeOnCPU = 0.0f;

	oldImagePixArr = (Pixel*)malloc(size * sizeof(Pixel));
	newImagePixArr = (Pixel*)malloc(size * sizeof(Pixel));
	highlightedChangePixArr = (Pixel*)malloc(size * sizeof(Pixel));

	convertBitmapToPixelArr(oldImagePixArr, oldImage->bitmap, size);
	convertBitmapToPixelArr(newImagePixArr, newImage->bitmap, size);

	
	StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	CPUChangeDetection(oldImagePixArr, newImagePixArr, highlightedChangePixArr, threshold, oldImage->width, oldImage->height);

	sdkStopTimer(&timer);
	timeOnCPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
	
	cout << "Time Taken on CPU : " << timeOnCPU << " ms" << endl;

	convertPixelArrToBitmap(detectedChanges, highlightedChangePixArr, size);

	cpu_cleanup();
}

void cpu_cleanup(void)
{
	if (highlightedChangePixArr)
	{
		free(highlightedChangePixArr);
		highlightedChangePixArr = NULL;
	}

	if (newImagePixArr)
	{
		free(newImagePixArr);
		newImagePixArr = NULL;
	}

	if (oldImagePixArr)
	{
		free(oldImagePixArr);
		oldImagePixArr = NULL;
	}
}
