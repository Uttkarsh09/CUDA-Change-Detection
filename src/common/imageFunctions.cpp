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


string mapIDToImageFormatName(FREE_IMAGE_FORMAT id)
{
	string fif = FreeImage_GetFormatFromFIF(id);
	return (fif == "") ? "!!! FIF_UNKNOWN !!!" : fif;
}


string mapIDToColorTypeName(FREE_IMAGE_COLOR_TYPE id)
{
	switch(id)
	{
		case 0: return "FIC_MINISWHITE";
		case 1: return "FIC_MINISBLACK";
		case 2: return "FIC_RGB       ";
		case 3: return "FIC_PALETTE   ";
		case 4: return "FIC_RGBALPHA  ";
		case 5: return "FIC_CMYK      ";
		default: return "FIF_UNKNOWN";
	}
}


void printImageData(ImageData *image)
{
	cout << "Address \t= " << image->address << endl;
	cout << "Resolution \t= " << image->width << "x" << image->height << endl;
	cout << "Bits Per Pixel \t= " << image->bpp << endl;
	cout << "Color Type \t= " << image->colorType << " -> " << mapIDToColorTypeName(image->colorType) << endl;
	cout << "Pitch \t\t=" << image->pitch << endl;
	cout << "Bitmap Width \t=" << image->bitmapWidth << endl;
	cout << "Image Format \t= " << image->imageFormat << " -> " << mapIDToImageFormatName(image->imageFormat) << endl;
	cout << endl;
}


void populateImageData(ImageData *imageData)
{
	imageData->width = FreeImage_GetWidth(imageData->dib);
	imageData->height = FreeImage_GetHeight(imageData->dib);
 	imageData->bpp = FreeImage_GetBPP(imageData->dib);	
	imageData->colorType = FreeImage_GetColorType(imageData->dib); 
	imageData->pitch = FreeImage_GetPitch(imageData->dib);
	imageData->bitmapWidth = FreeImage_GetLine(imageData->dib);
	imageData->imageFormat = FreeImage_GetFileType(imageData->address.c_str(), 0);

	if(imageData->imageFormat == FIF_UNKNOWN)
	{
		imageData->imageFormat = FreeImage_GetFIFFromFilename(imageData->address.c_str());
	
		if(imageData->imageFormat == FIF_UNKNOWN)
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
	
	cout << endl << "Image Saved Successfully at " << address << endl;
}


void convertBitmapToPixelArr(Pixel *pixelArr, uint8_t *bitmap, size_t size)
{
	uint8_t *bitmapPtrCpy = bitmap;
	
	for(int i=0 ; i<size ; i++, bitmapPtrCpy+=3)
	{
		pixelArr[i].blue = bitmapPtrCpy[0];
		pixelArr[i].green = bitmapPtrCpy[1];
		pixelArr[i].red = bitmapPtrCpy[2];
	}
}


void convertPixelArrToBitmap(uint8_t *bitmap, Pixel *pixelArr, size_t size)
{
	uint8_t *bitmapPtrCpy = bitmap;

	for(int i=0 ; i<size ; i++, bitmapPtrCpy+=3)
	{
		bitmapPtrCpy[0] = pixelArr[i].blue;
		bitmapPtrCpy[1] = pixelArr[i].green;
		bitmapPtrCpy[2] = pixelArr[i].red;
	}
}