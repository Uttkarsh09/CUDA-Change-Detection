cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(CUDA_CHANGE_DETECTION)
enable_language(CUDA)

# set(CMAKE_CXX_FLAGS "-Wall")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/lib)
set(INCLUDE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/../include/)


if(UNIX)
	message("~~~~~~~~~~ BUILDING FOR UNIX SYSTEMS ~~~~~~~~~~")
elseif(WIN32)
	message("~~~~~~~~~~ BUILDING FOR WINDOWS 32bit ~~~~~~~~~~")
endif()


add_library(machineInformation STATIC machineInformation.cu)
# target_include_directories(machineInformation PUBLIC ${INCLUDE_FOLDER})

add_executable(main main.cu)
target_include_directories(main PUBLIC ${INCLUDE_FOLDER})

target_link_libraries(main machineInformation)



# message("this is the CMAKE_CURRENT_SOURCE_DIR -> ${CMAKE_CURRENT_SOURCE_DIR}")
# message("this is the CMAKE_BINARY_DIR -> ${CMAKE_BINARY_DIR}")
# message("this is the CMAKE_INCLUDE_PATH -> ${CMAKE_INCLUDE_PATH}")
# message("this is the CMAKE_SOURCE_DIR -> ${CMAKE_SOURCE_DIR}")
# message("this is the CMAKE_CURRENT_BINARY_DIR -> ${CMAKE_CURRENT_BINARY_DIR}")
# message("this is the CMAKE_CURRENT_FUNCTION_LIST_DIR -> ${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
# message("this is the CMAKE_HOME_DIRECTORY -> ${CMAKE_HOME_DIRECTORY}")
# message("this is the INCLUDE_FOLDER -> ${INCLUDE_FOLDER}")
