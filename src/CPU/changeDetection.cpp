#include "../../include/CPU/changeDetection.hpp"

void runOnCPU()
{
	FreeImage_SetOutputMessage(FreeImageErrorHandler);
	
	IMAGE_DATA oldImage, newImage, oldGrayImage, newGrayImage;
	oldImage.address = getImage("old.png");
	newImage.address = getImage("new.png");

	oldImage.dib = ImageFormatIndependentLoader(oldImage.address.c_str(), 0);
	newImage.dib = ImageFormatIndependentLoader(newImage.address.c_str(), 0);
	
	populateImageData(&oldImage);
	populateImageData(&newImage);
	printImageData(oldImage);
	printImageData(newImage);
	
	oldGrayImage.dib = FreeImage_ConvertToGreyscale(oldImage.dib);
	oldGrayImage.address = getImage("oldGray.png");
	populateImageData(&oldGrayImage);
	saveImage(oldGrayImage);
	printImageData(oldGrayImage);

	newGrayImage.dib = FreeImage_ConvertToGreyscale(newImage.dib);
	newGrayImage.address = getImage("newGray.png");
	populateImageData(&newGrayImage);
	saveImage(newGrayImage);
	printImageData(oldGrayImage);	


	IMAGE_DATA highlightedChanges;
	copyImage(&highlightedChanges, &oldImage);
	highlightedChanges.address = getImage("highlightedChanges.png");
	
	convertToRGBGreyscale(&highlightedChanges, &oldGrayImage);

	FIBITMAP *differences = detectChanges(oldGrayImage, newGrayImage);
	highlightChangesInImage(&highlightedChanges, differences);
	saveImage(highlightedChanges);


	FreeImage_Unload(oldImage.dib);
	FreeImage_Unload(newImage.dib);
	FreeImage_Unload(oldGrayImage.dib);
	FreeImage_Unload(newGrayImage.dib);

}
