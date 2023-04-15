#pragma once

// ~~~~~~~~ Common ~~~~~~~~
#include <iostream>
#include <string>
#include "dataTypes.hpp"
#include "FreeImage.h"
#include "subRoutines.hpp"


// ~~~~~~~~ CPU ~~~~~~~~
#include "imageOperations.hpp"
#include "changeDetection.hpp"


// ~~~~~~~~ CUDA ~~~~~~~~
#include "cudaChangeDetection.cuh"
#include "deviceInformation.cuh"


// ~~~~~~~~ OpenCL ~~~~~~~~


using namespace std;