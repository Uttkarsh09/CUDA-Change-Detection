#pragma once

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    #define PLATFORM 1
#elif defined(__linux__)
    #define PLATFORM 2
#elif defined(__APPLE__)
    #define PLATFORM 3
#endif

#if defined(__NVCC__)
    #define HPP 1
#else
    #define HPP 2
#endif
