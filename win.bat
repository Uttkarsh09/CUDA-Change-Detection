
cls

@REM GPU Detection
@echo off
where /q nvcc
if ERRORLEVEL 1 (set hpp=OpenCL) else (set hpp=CUDA)

if %hpp%==CUDA (
    
    nvcc.exe -c -o "./bin/cudaChangeDetection.obj" src/GPU/CUDA/cudaChangeDetection.cu
    cd "./bin/"
    cl.exe /c /EHsc /std:c++20 /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\include" "../src/main.cpp" "../src/common/imageFunctions.cpp" "../src/CPU/changeDetection.cpp" "../src/CPU/imageOperations.cpp"
    link.exe /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\lib\x64" /LIBPATH:"../lib" main.obj imageFunctions.obj changeDetection.obj imageOperations.obj cudaChangeDetection.obj
    @copy main.exe "../" > nul
    cd ../
    main.exe

) else (

    cd "./bin/"
    cl.exe /c /EHsc /std:c++20 /I "C:\OCL_SDK_Light\include" "../src/main.cpp" "../src/GPU/OpenCL/openclChangeDetection.cpp" "../src/common/imageFunctions.cpp" "../src/CPU/changeDetection.cpp" "../src/CPU/imageOperations.cpp"
    link.exe /LIBPATH:"C:\OCL_SDK_Light\lib\x86_64" /LIBPATH:"../lib" main.obj openclChangeDetection.obj imageFunctions.obj changeDetection.obj imageOperations.obj
    @copy main.exe "../" > nul
    cd ../
    main.exe

)
