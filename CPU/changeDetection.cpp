#include<iostream>
#include<FreeImage.h>
#include<string>

using namespace std;

struct RGB_TO_GREYSCALE{
	float red = 0.3, green = 0.59, blue = 0.11;
};

typedef struct Image_Metadata{
	int width;
	int height;
	int bpp; // ? Bits Per Pixel
	unsigned int memorySize;	// ? -> values can be a standard approximation, may vary between using different C++ standard libs
	string address;
	FREE_IMAGE_FORMAT imageFormat;
	FIBITMAP *dib;
	FREE_IMAGE_COLOR_TYPE colorType; // ? One of -> FIC_MINISWHITE-0 | FIC_MINISBLACK-1 | FIC_RGB-2 | FIC_PALETTE-3 | FIC_RGBALPHA-4 | FIC_CMYK-5	

	// void copyImageMetadata(struct Image_Metadata img, string address){
	// 	this->width = img.width;
	// 	FreeImage_GetWidth
	// 	this->height = img.height;
	// 	this->bpp = FreeImage_GetBPP(this->dib);
	// 	this->address = address;
	// 	this->imageFormat = FreeImage_GetFileType(this->address.c_str(), 0);
	// }
} IMAGE_DATA;


// * lpsz -> Long pointer to Null Terminated String
// * dib -> device independent bitmap
FIBITMAP* ImageFormatIndependentLoader(const char* lpszPathName, int flag){
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;

	// ? The second argument is not used by FreeImgae!!
	fif = FreeImage_GetFileType(lpszPathName, 0);

	// ? this means there is no signature, try to guess it from the file name (.png, ...)
	if(fif == FIF_UNKNOWN)
		fif = FreeImage_GetFIFFromFilename(lpszPathName);

	if((fif != FIF_UNKNOWN) && FreeImage_FIFSupportsReading(fif)){
		FIBITMAP *dib = FreeImage_Load(fif, lpszPathName, flag);
		return dib;
	}
	
	// ? fif == FIF_UNKNOWN, so we terminate
	cout << "ERROR -> FILE IMAGE FORMAT UNKNOWN";
	exit(1);
}


void FreeImageErrorHandler(FREE_IMAGE_FORMAT fif, const char *message) {
	cout << endl << "***";
	if(fif != FIF_UNKNOWN) {
		printf("%s Format\n", FreeImage_GetFormatFromFIF(fif));
	}
	cout << message << endl; 
	cout << endl << "***";
}


void populateImageMetadata(IMAGE_DATA *imgData){
	imgData->width = FreeImage_GetWidth(imgData->dib);
	imgData->height = FreeImage_GetHeight(imgData->dib);
 	imgData->bpp = FreeImage_GetBPP(imgData->dib);
	if(!imgData->address.compare("")){
		cout << "No address provided, can't get FIF";
		exit(1);
	}
	imgData->imageFormat = FreeImage_GetFileType(imgData->address.c_str(), 0);
	imgData->memorySize = FreeImage_GetMemorySize(imgData->dib);	
	imgData->colorType = FreeImage_GetColorType(imgData->dib); 
}


string mapIDToImageFormatName(FREE_IMAGE_FORMAT id){
	switch (id)
	{
		case 0: return "FIF_BMP";
		case 1: return "FIF_ICO";
		case 2: return "FIF_JPEG";
		case 3: return "FIF_JNG";
		case 4: return "FIF_KOALA";
		case 5: return "FIF_LBM or FIF_IFF";
		case 6: return "FIF_MNG";
		case 7: return "FIF_PBM";
		case 8: return "FIF_PBMRAW";
		case 9: return "FIF_PCD";
		case 10: return "FIF_PCX";
		case 11: return "FIF_PGM";
		case 12: return "FIF_PGMRAW";
		case 13: return "FIF_PNG";
		case 14: return "FIF_PPM";
		case 15: return "FIF_PPMRAW";
		case 16: return "FIF_RAS";
		case 17: return "FIF_TARGA";
		case 18: return "FIF_TIFF";
		case 19: return "FIF_WBMP";
		case 20: return "FIF_PSD";
		case 21: return "FIF_CUT";
		case 22: return "FIF_XBM";
		case 23: return "FIF_XPM";
		case 24: return "FIF_DDS";
		case 25: return "FIF_GIF";
		case 26: return "FIF_HDR";
		case 27: return "FIF_FAXG3";
		case 28: return "FIF_SGI";
		case 29: return "FIF_EXR";
		case 30: return "FIF_J2K";
		case 31: return "FIF_JP2";
		case 32: return "FIF_PFM";
		case 33: return "FIF_PICT";
		case 34: return "FIF_RAW";
		case 35: return "FIF_WEBP";
		case 36: return "FIF_JXR";
	default:
		return "FIF_UNKNOWN";
	}
}


string mapIDToColorTypeName(FREE_IMAGE_COLOR_TYPE id){
	switch(id){
		case 0: return "FIC_MINISWHITE";
		case 1: return "FIC_MINISBLACK";
		case 2: return "FIC_RGB       ";
		case 3: return "FIC_PALETTE   ";
		case 4: return "FIC_RGBALPHA  ";
		case 5: return "FIC_CMYK      ";
		default: return "FIF_UNKNOWN";
	}
}


void printImageData(IMAGE_DATA img){
	cout << "Address \t= " << img.address << endl;
	cout << "Resolution \t= " << img.width << "x" << img.height << endl;
	cout << "Bits Per Pixel \t= " << img.bpp << endl;
	cout << "ImageFormat \t= " << img.imageFormat << " -> " << FreeImage_GetFormatFromFIF(img.imageFormat) << endl;
	cout << "MemorySize \t= " << img.memorySize << endl;
	cout << "ColorType \t= " << img.colorType << " -> " << mapIDToColorTypeName(img.colorType) << endl;
	cout << endl;
}

int main(){
	FreeImage_Initialise();
	FreeImage_SetOutputMessage(FreeImageErrorHandler);
	
	IMAGE_DATA oldImage, newImage;
	oldImage.address = "./images/old1.png";
	newImage.address = "./images/new1.png";


	oldImage.dib = ImageFormatIndependentLoader(oldImage.address.c_str(), 0);
	newImage.dib = ImageFormatIndependentLoader(newImage.address.c_str(), 0);
	
	populateImageMetadata(&oldImage);
	populateImageMetadata(&newImage);
	printImageData(oldImage);
	printImageData(newImage);

	IMAGE_DATA oldGrayImage, newGrayImage;
	oldGrayImage.dib = FreeImage_ConvertToGreyscale(oldImage.dib);
	newGrayImage.dib = FreeImage_ConvertToGreyscale(newImage.dib);
	oldGrayImage.address = "./images/oldGray.png";
	newGrayImage.address = "./images/newGray.png";

	cout<<"HERE"<<endl;
	
	populateImageMetadata(&oldGrayImage);
	populateImageMetadata(&newGrayImage);

	printImageData(oldGrayImage);
	printImageData(newGrayImage);

	if(FreeImage_Save(oldGrayImage.imageFormat, oldGrayImage.dib, oldGrayImage.address.c_str(), 0)){
		cout << "Image Saved Successfully" << endl;
	}
	else {
		char err[500];
        sprintf(err, "Can't save the file");
		printf("\n~~~~~~~~~~\n");
        perror(err);
		printf("~~~~~~~~~~\n");
        exit(1);
	}

	if(FreeImage_Save(newGrayImage.imageFormat, newGrayImage.dib, newGrayImage.address.c_str(), 0)){
		cout << "Saved successfull at " << newGrayImage.address << endl;
	}
	else {
		char err[500];
        sprintf(err, "Can't save the file");
		printf("~~~~~~~~~~\n");
        perror(err);
		printf("~~~~~~~~~~\n");
        exit(1);
	}
	


	FreeImage_Unload(oldImage.dib);
	FreeImage_DeInitialise();
	return 0;
}