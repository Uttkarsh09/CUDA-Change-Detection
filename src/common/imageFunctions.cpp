#include "../../include/common/imageFunctions.hpp"

string getImage(string image)
{
	string imageDirectory;
	filesystem::path currentPath = filesystem::current_path();
	
	if (PLATFORM == 1) 					
		imageDirectory = "images\\";	// Windows
	else
		imageDirectory = "images/";		// Linux and macOS
	
	filesystem::path imagePath =  currentPath / imageDirectory / image;

	return imagePath.string();
}

void FreeImageErrorHandler(FREE_IMAGE_FORMAT fif, const char *message)
{
	cout << endl << "***";
	if(fif != FIF_UNKNOWN)
		cout << FreeImage_GetFormatFromFIF(fif) << "Format" << endl;
	cout << message << endl; 
	cout << endl << "***";
}
