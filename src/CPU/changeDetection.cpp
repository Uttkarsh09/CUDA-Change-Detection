#include "../../include/CPU/changeDetection.hpp"

void runOnCPU()
{
	FreeImage_SetOutputMessage(FreeImageErrorHandler);
	
	IMAGE_DATA oldImage, newImage, oldGreyImage, newGreyImage;
	oldImage.address = getImagePath("old.png");
	oldImage.dib = ImageFormatIndependentLoader(oldImage.address.c_str(), 0);

	newImage.address = getImagePath("new.png");
	newImage.dib = ImageFormatIndependentLoader(newImage.address.c_str(), 0);
	
	populateImageData(&oldImage);
	printImageData(oldImage);
	
	populateImageData(&newImage);
	printImageData(newImage);
	
	oldGreyImage.dib = FreeImage_Allocate(oldImage.width, oldImage.height, 8);
	oldGreyImage.address = getImagePath("oldGrey.png");
	populateImageData(&oldGreyImage); 
	convertTo8bitGreyscale(&oldGreyImage, &oldImage);
	saveImage(oldGreyImage);
	printImageData(oldGreyImage);

	newGreyImage.dib = FreeImage_Allocate(newImage.width, newImage.height, 8);
	newGreyImage.address = getImagePath("newGrey.png");
	populateImageData(&newGreyImage);
	convertTo8bitGreyscale(&newGreyImage, &newImage);
	saveImage(newGreyImage);
	printImageData(oldGreyImage);	


	IMAGE_DATA highlightedChanges;
	copyImage(&highlightedChanges, &oldImage);
	highlightedChanges.address = getImagePath("highlightedChanges.png");
	
	convertTo24bitGreyscale(&highlightedChanges, &oldGreyImage);

	FIBITMAP *differences = detectChanges(oldGreyImage, newGreyImage);
	highlightChangesInImage(&highlightedChanges, differences);
	saveImage(highlightedChanges);


	FreeImage_Unload(oldImage.dib);
	FreeImage_Unload(newImage.dib);
	FreeImage_Unload(oldGreyImage.dib);
	FreeImage_Unload(newGreyImage.dib);
}
