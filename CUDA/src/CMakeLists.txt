cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(CUDA_CHANGE_DETECTION)
enable_language(CUDA)
# enable_language(C)

# set(CMAKE_CXX_FLAGS "-Wall")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/lib)
set(INCLUDE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/../include/)
set(LIBRARY_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/../libraries/)


add_library(machineInformation STATIC machineInformation.cu)

# add_library(freeImage SHARED ${LIBRARY_FOLDER}/FreeImage/Source/)
# target_include_directories(freeImage PUBLIC ${LIBRARY_FOLDER}/FreeImage/Source/)

include_directories(${LIBRARY_FOLDER}/FreeImage)
add_subdirectory(${LIBRARY_FOLDER}/FreeImage/Source)

add_executable(main main.cu)
target_include_directories(main PUBLIC ${INCLUDE_FOLDER})

target_link_libraries(main machineInformation)
target_link_libraries(main freeImage)
# target_link_libraries(main freeImage)


# message("LIBRARY FOLDER = ${LIBRARY_FOLDER}")
# message("this is the CMAKE_CURRENT_SOURCE_DIR -> ${CMAKE_CURRENT_SOURCE_DIR}")
# message("this is the CMAKE_BINARY_DIR -> ${CMAKE_BINARY_DIR}")
# message("this is the CMAKE_INCLUDE_PATH -> ${CMAKE_INCLUDE_PATH}")
# message("this is the CMAKE_SOURCE_DIR -> ${CMAKE_SOURCE_DIR}")
# message("this is the CMAKE_CURRENT_BINARY_DIR -> ${CMAKE_CURRENT_BINARY_DIR}")
# message("this is the CMAKE_CURRENT_FUNCTION_LIST_DIR -> ${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
# message("this is the CMAKE_HOME_DIRECTORY -> ${CMAKE_HOME_DIRECTORY}")
# message("this is the INCLUDE_FOLDER -> ${INCLUDE_FOLDER}")
