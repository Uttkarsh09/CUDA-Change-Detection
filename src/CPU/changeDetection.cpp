#include "../../include/CPU/changeDetection.hpp"

void runOnCPU(ImageData *oldImage, ImageData *newImage, int threshold, uint8_t *detectedChanges)
{
	Pixel *oldImagePixArr, *newImagePixArr, *highlightedChangePixArr;
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
	
	cout << "Time Taken on CPU: " << timeOnCPU << "ms" << endl;

	convertPixelArrToBitmap(detectedChanges, highlightedChangePixArr, size);

	free(oldImagePixArr);
	free(newImagePixArr);
	free(highlightedChangePixArr);
}