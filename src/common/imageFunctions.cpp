#include "../../include/common/imageFunctions.hpp"

string getImagePath(string imageName)
{
	string imageDirectory;
	filesystem::path currentPath = filesystem::current_path();
	
	if (PLATFORM == 1) 					
		imageDirectory = "images\\";	// Windows
	else
		imageDirectory = "images/";		// Linux and macOS
	
	filesystem::path imagePath =  currentPath / imageDirectory / imageName;

	return imagePath.string();
}


void FreeImageErrorHandler(FREE_IMAGE_FORMAT fif, const char *message)
{
	cout <<  "***" << endl;
	if(fif != FIF_UNKNOWN)
		cout << FreeImage_GetFormatFromFIF(fif) << "Format" << endl;
	cout << message << endl; 
	cout << "***" << endl;
}


// * lpsz -> Long pointer to Null Terminated String
// * dib -> device independent bitmap
FIBITMAP* imageFormatIndependentLoader(const char* lpszPathName, int flag)
{
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;

	// ? The second argument is not used by FreeImgae!!
	fif = FreeImage_GetFileType(lpszPathName, 0);

	// ? this means there is no signature, try to guess it from the file name (.png, ...)
	if(fif == FIF_UNKNOWN)
		fif = FreeImage_GetFIFFromFilename(lpszPathName);

	if((fif != FIF_UNKNOWN) && FreeImage_FIFSupportsReading(fif))
	{
		FIBITMAP *dib = FreeImage_Load(fif, lpszPathName, flag);
		return dib;
	}
	
	// ? fif == FIF_UNKNOWN, so we terminate
	cout << "ERROR -> FILE IMAGE FORMAT UNKNOWN";
	exit(1);
}


void populateImageData(IMAGE_DATA *imgData)
{
	imgData->width = FreeImage_GetWidth(imgData->dib);
	imgData->height = FreeImage_GetHeight(imgData->dib);
 	imgData->bpp = FreeImage_GetBPP(imgData->dib);
	imgData->memorySize = FreeImage_GetMemorySize(imgData->dib);	
	imgData->colorType = FreeImage_GetColorType(imgData->dib); 
	imgData->bitmapWidth = FreeImage_GetPitch(imgData->dib);

	imgData->imageFormat = FreeImage_GetFileType(imgData->address.c_str(), 0);

	if(imgData->imageFormat == FIF_UNKNOWN)
	{
		imgData->imageFormat = FreeImage_GetFIFFromFilename(imgData->address.c_str());
	
		if(imgData->imageFormat == FIF_UNKNOWN)
		{
			cout << "ERROR: Can't get FIF (Free Image Format)";
			exit(1);
		}
	}
}


void saveImage(FIBITMAP *dib, FREE_IMAGE_FORMAT imageFormat, string address)
{
	bool saved = FreeImage_Save(imageFormat, dib, address.c_str(), 0);

	if(!saved)
	{
		cout << endl << "~~~~~~~~~~" << endl;
		perror("Can't save the file");
		cout << endl << "~~~~~~~~~~" << endl;
		exit(1);
	} 
	
	cout << "Image Saved Successfully at " << address << endl;
}