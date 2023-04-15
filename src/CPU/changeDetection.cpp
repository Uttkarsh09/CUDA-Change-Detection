#include "../../include/CPU/changeDetection.hpp"

void runOnCPU()
{
	FreeImage_SetOutputMessage(FreeImageErrorHandler);
	
	IMAGE_DATA oldImage, newImage, oldGreyImage, newGreyImage;
	oldImage.address = getImage("old.png");
	newImage.address = getImage("new.png");

	oldImage.dib = ImageFormatIndependentLoader(oldImage.address.c_str(), 0);
	newImage.dib = ImageFormatIndependentLoader(newImage.address.c_str(), 0);
	
	populateImageData(&oldImage);
	populateImageData(&newImage);
	printImageData(oldImage);
	printImageData(newImage);
	
	oldGreyImage.dib = FreeImage_ConvertToGreyscale(oldImage.dib);
	oldGreyImage.address = getImage("oldGrey.png");
	populateImageData(&oldGreyImage);
	saveImage(oldGreyImage);
	printImageData(oldGreyImage);

	newGreyImage.dib = FreeImage_ConvertToGreyscale(newImage.dib);
	newGreyImage.address = getImage("newGrey.png");
	populateImageData(&newGreyImage);
	saveImage(newGreyImage);
	printImageData(oldGreyImage);	


	IMAGE_DATA highlightedChanges;
	copyImage(&highlightedChanges, &oldImage);
	highlightedChanges.address = getImage("highlightedChanges.png");
	
	convertTo24bitGreyscale(&highlightedChanges, &oldGreyImage);

	FIBITMAP *differences = detectChanges(oldGreyImage, newGreyImage);
	highlightChangesInImage(&highlightedChanges, differences);
	saveImage(highlightedChanges);


	FreeImage_Unload(oldImage.dib);
	FreeImage_Unload(newImage.dib);
	FreeImage_Unload(oldGreyImage.dib);
	FreeImage_Unload(newGreyImage.dib);
}
