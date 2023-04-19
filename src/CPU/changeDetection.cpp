#include "../../include/CPU/changeDetection.hpp"

void runOnCPU(ImageData *oldImage, ImageData *newImage, int threshold, uint8_t *detectedChanges)
{
	uint8_t *highlightChangesBitmap, *startCpy;
	FIBITMAP *highlightChangesDib;

	highlightChangesBitmap = (uint8_t*)malloc(oldImage->height * oldImage->pitch);
	startCpy = highlightChangesBitmap;
	
	auto start = std::chrono::high_resolution_clock::now();
	CPUChangeDetection(
		oldImage->bitmap,
		newImage->bitmap,
		detectedChanges,
		oldImage->pitch,
		oldImage->width,
		oldImage->height,
		threshold
	);
	auto stop = std::chrono::high_resolution_clock::now();
	
	auto CPU_Duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	cout << "CPU Duration = " << CPU_Duration.count() << endl;
}