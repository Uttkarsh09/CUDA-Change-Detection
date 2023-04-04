#include<iostream>
#include<FreeImage.h>
#include<string>

using namespace std;

struct RGB_TO_GREYSCALE{
	float red = 0.3, green = 0.59, blue = 0.11;
};

typedef struct Image_Metadata{
	int width=-1;
	int height=-1;
	int bpp=-1; // ? Bits Per Pixel
	unsigned int memorySize = 0;					// ? -> values can be a standard approximation, may vary between using different C++ standard libs
	string address = "unknown";
	FREE_IMAGE_FORMAT imageFormat = FIF_UNKNOWN;
	FIBITMAP *dib;
	FREE_IMAGE_COLOR_TYPE colorType; 				// ? One of -> FIC_MINISWHITE-0 | FIC_MINISBLACK-1 | FIC_RGB-2 | FIC_PALETTE-3 | FIC_RGBALPHA-4 | FIC_CMYK-5	
} IMAGE_DATA;

string mapIDToColorTypeName(FREE_IMAGE_COLOR_TYPE);
FIBITMAP* ImageFormatIndependentLoader(const char*, int);
void FreeImageErrorHandler(FREE_IMAGE_FORMAT, const char*);
void printImageData(IMAGE_DATA);
string mapIDToImageFormatName(FREE_IMAGE_FORMAT);
string mapIDToColorTypeName(FREE_IMAGE_COLOR_TYPE);
void populateImageData(IMAGE_DATA*);


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


void FreeImageErrorHandler(FREE_IMAGE_FORMAT fif, const char *message){
	cout << endl << "***";
	if(fif != FIF_UNKNOWN) {
		printf("%s Format\n", FreeImage_GetFormatFromFIF(fif));
	}
	cout << message << endl; 
	cout << endl << "***";
}


string mapIDToImageFormatName(FREE_IMAGE_FORMAT id){
	string fif = FreeImage_GetFormatFromFIF(id);
	return (fif == "") ? "!!! FIF_UNKNOWN !!!" : fif;
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
	cout << "Memory Size \t= " << img.memorySize << endl;
	cout << "Color Type \t= " << img.colorType << " -> " << mapIDToColorTypeName(img.colorType) << endl;
	cout << "Image Format \t= " << img.imageFormat << " -> " << mapIDToImageFormatName(img.imageFormat) << endl;
	cout << endl;
}


void saveImage(IMAGE_DATA imgData, string address=""){
	if(address == "")
	imgData.address = address;
	bool saved = FreeImage_Save(imgData.imageFormat, imgData.dib, imgData.address.c_str(), 0);
	if(!saved){
		cout << endl << "~~~~~~~~~~" << endl;
		perror("Can't save the file");
		cout << endl << "~~~~~~~~~~" << endl;
		exit(1);
	} 
	
	cout << "Image Saved Successfully at " << imgData.address << endl;
}


void populateImageData(IMAGE_DATA *imgData){
	imgData->width = FreeImage_GetWidth(imgData->dib);
	imgData->height = FreeImage_GetHeight(imgData->dib);
 	imgData->bpp = FreeImage_GetBPP(imgData->dib);
	imgData->memorySize = FreeImage_GetMemorySize(imgData->dib);	
	imgData->colorType = FreeImage_GetColorType(imgData->dib); 

	imgData->imageFormat = FreeImage_GetFileType(imgData->address.c_str(), 0);

	if(imgData->imageFormat == FIF_UNKNOWN){
		imgData->imageFormat = FreeImage_GetFIFFromFilename(imgData->address.c_str());
	
		if(imgData->imageFormat == FIF_UNKNOWN){
			cout << "Can't get FIF (Free Image Format)";
			exit(1);
		}
	}
}

int main(){
	FreeImage_Initialise();
	FreeImage_SetOutputMessage(FreeImageErrorHandler);
	
	IMAGE_DATA oldImage, newImage, oldGrayImage, newGrayImage;
	oldImage.address = "./images/old1.png";
	newImage.address = "./images/new1.png";

	oldImage.dib = ImageFormatIndependentLoader(oldImage.address.c_str(), 0);
	newImage.dib = ImageFormatIndependentLoader(newImage.address.c_str(), 0);
	
	populateImageData(&oldImage);
	populateImageData(&newImage);

	printImageData(oldImage);
	printImageData(newImage);
	
	oldGrayImage.dib = FreeImage_ConvertToGreyscale(oldImage.dib);
	oldGrayImage.address = "./images/oldGray.png";
	populateImageData(&oldGrayImage);
	saveImage(oldGrayImage, oldGrayImage.address);
	printImageData(oldGrayImage);

	newGrayImage.dib = FreeImage_ConvertToGreyscale(newImage.dib);
	newGrayImage.address = "./images/newGray.png";
	populateImageData(&newGrayImage);
	saveImage(newGrayImage, newGrayImage.address);
	printImageData(newGrayImage);

	FreeImage_Unload(oldImage.dib);
	FreeImage_Unload(newImage.dib);
	FreeImage_Unload(oldGrayImage.dib);
	FreeImage_Unload(newGrayImage.dib);
	FreeImage_DeInitialise();
	return 0;
}