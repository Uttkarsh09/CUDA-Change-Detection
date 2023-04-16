#pragma once

#include <iostream>
#include <filesystem>
#include <string>
using namespace std;

#include "systemMacros.hpp"
#include "FreeImage.h"

#if (PLATFORM == 1)
    #pragma comment(lib, "FreeImage.lib")
#endif

string getImagePath(string);
void FreeImageErrorHandler(FREE_IMAGE_FORMAT fif, const char *message);
