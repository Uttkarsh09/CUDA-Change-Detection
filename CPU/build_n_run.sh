# ONLY FOR GNU LINUX
# clear
# g++ changeDetection.cpp -o cd -lfreeimage
# ./cd

clear

echo "compiling src files"
cd ./bin
g++ -I ../include/ -c ../src/*.cpp
cd ..

echo "creating staticly linked library"
ar rs ./bin/libimgoperations.a ./bin/imageOperations.o

echo "creating executable"
g++ -o ./bin/detectChanges -L ./bin/ ./bin/changeDetection.o -limgoperations -lfreeimage

echo "RUNNING..."
./bin/detectChanges