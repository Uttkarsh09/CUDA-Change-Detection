#include "../../include/CPU/changeDetection.hpp"

void runOnCPU(ImageData *oldImage, ImageData *newImage, int threshold, uint8_t *detectedChanges)
{
	Pixel *oldImagePixArr, *newImagePixArr, *highlightedChangePixArr;
	uint8_t *highlightChangesBitmap, *startCpy;
	FIBITMAP *highlightChangesDib;
	size_t size = (oldImage->height * oldImage->pitch)/3;

	oldImagePixArr = (Pixel*)malloc(size * sizeof(Pixel));
	newImagePixArr = (Pixel*)malloc(size * sizeof(Pixel));
	highlightedChangePixArr = (Pixel*)malloc(size * sizeof(Pixel));

	convertBitmapToPixelArr(oldImagePixArr, oldImage->bitmap, size);
	convertBitmapToPixelArr(newImagePixArr, newImage->bitmap, size);

	
	// ! auto start = std::chrono::high_resolution_clock::now();

	CPUChangeDetection(oldImagePixArr, newImagePixArr, highlightedChangePixArr, threshold, oldImage->width, oldImage->height);

	// ! auto stop = std::chrono::high_resolution_clock::now();
	
	// ! auto CPU_Duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	// ! cout << "CPU Duration = " << CPU_Duration.count() << endl;

	convertPixelArrToBitmap(detectedChanges, highlightedChangePixArr, size);

	free(oldImagePixArr);
	free(newImagePixArr);
	free(highlightedChangePixArr);
}