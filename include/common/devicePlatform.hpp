#pragma once

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    
    #if defined(__NVCC__) 
        #include "../GPU/CUDA/cudaChangeDetection.cuh"
    #else
        #include "../GPU/OpenCL/openclChangeDetection.hpp"
    #endif

#elif defined(__linux__)

    #if defined(__NVCC__)
        #include "../GPU/CUDA/cudaChangeDetection.cuh"
    #else
        #include "../GPU/OpenCL/openclChangeDetection.hpp"
    #endif

#elif defined(__APPLE__)

    #include "../GPU/OpenCL/openclChangeDetection.hpp"

#endif
