#include <iostream>
#include "FreeImage.h"

using namespace std;

void FreeImageErrorHandler(FREE_IMAGE_FORMAT fif, const char *message){
	cout << endl << "***";
	if(fif != FIF_UNKNOWN) {
		printf("%s Format\n", FreeImage_GetFormatFromFIF(fif));
	}
	cout << message << endl; 
	cout << endl << "***";
}
