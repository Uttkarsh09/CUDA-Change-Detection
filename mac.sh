clear

cd "./bin/"

clang++ -Wno-deprecated-declarations -std=c++20 -o main "../src/main.cpp" "../src/GPU/OpenCL/openclChangeDetection.cpp" "../src/common/imageFunctions.cpp" "../src/CPU/changeDetection.cpp" "../src/CPU/imageOperations.cpp" -framework OpenCL -lFreeImage

mv main "../"

cd ../

./main 1024 1
./main 2048 0
./main 4096 0
./main 8192 0
./main 10000 0