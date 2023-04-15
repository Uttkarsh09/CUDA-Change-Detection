#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    
    #define PLATFORM 1
    
    #if defined(__NVCC__) 
        #include "../GPU/CUDA/cudaProperties.cuh"
    #else
        #include "../GPU/OpenCL/openclProperties.hpp"
    #endif

#elif defined(__linux__)

    #define PLATFORM 2

    #if defined(__NVCC__)
        #include "../GPU/CUDA/cudaProperties.cuh"
    #else
        #include "../GPU/OpenCL/openclProperties.hpp"
    #endif

#elif defined(__APPLE__)

    #define PLATFORM 3
    #include "../GPU/OpenCL/openclProperties.hpp"

#endif
