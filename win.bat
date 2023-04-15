
cls

@REM GPU Detection
@echo off
where /q nvcc
if ERRORLEVEL 1 (set hpp=OpenCL) else (set hpp=CUDA)

if %hpp%==CUDA (
    
    nvcc.exe -c -o cudaProperties.obj src/GPU/CUDA/cudaProperties.cu
    cl.exe /c /EHsc /std:c++20 /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\include" src/main.cpp src/common/imageFunctions.cpp src/CPU/changeDetection.cpp src/CPU/imageOperations.cpp
    link.exe /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\lib\x64" /LIBPATH:"./lib" Main.obj cudaProperties.obj imageFunctions.obj changeDetection.obj imageOperations.obj
    Main.exe

) else (

    cl.exe /c /EHsc /std:c++20 /I "C:\OCL_SDK_Light\include" /I ".\include" src/Main.cpp src/OpenCLProperties.cpp src/ImageFunctions.cpp
    link.exe /LIBPATH:"C:\OCL_SDK_Light\lib\x86_64" Main.obj OpenCLProperties.obj ImageFunctions.obj
    Main.exe

)
