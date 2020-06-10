cmake_minimum_required(VERSION 3.14)

#Cuda Setup
set(CMAKE_CUDA_STANDARD 14)
#NOT needed env variable set at:
#Settings -> Build, Execution, Deployment -> CMake
#set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

#CXX Setup
set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_C_COMPILER /usr/bin/gcc-7)
#set(CMAKE_C_COMPILER /usr/bin/clang-7)
set(CMAKE_CXX_COMPILER:FILEPATH /usr/bin/g++-8)
#set(CMAKE_CXX_COMPILER /usr/bin/clang++-7)  #clang++-8 NOT available in ubuntu 15 apr 2019

#set(GCC_EXTRA_COMPILE_FLAGS "-Ofast -fopenmp")
set(GCC_EXTRA_COMPILE_FLAGS "-O0 -g -fopenmp")
#set(GCC_EXTRA_LINK_FLAGS    "")
#NOT needed. Cuda target should add it
#but how to specify static or shared
#set(GCC_EXTRA_LINK_FLAGS    "-L/usr/local/cuda/lib64 -lcuda -lcudart")

#reset existing
set(CMAKE_CXX_FLAGS  ${GCC_EXTRA_COMPILE_FLAGS})
set(CMAKE_EXE_LINKER_FLAGS  ${GCC_EXTRA_LINK_FLAGS})

#add to existing
#set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_EXTRA_COMPILE_FLAGS}")
#set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_EXTRA_LINK_FLAGS}")

#Project name
#Must be after set compiler paths
project(04_wavelets CXX CUDA) #enable also CUDA

set(SRC "src")
set(EXAMPLES "examples")
set(TEST "test")
set(BIN "bin")

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../${BIN})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../${BIN})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../${BIN})

#include folders
set(INCLUDE "/home/hdaniel/Dropbox/01-libs/cpp")# "/usr/local/cuda-9.0/include")
include_directories(. ${INCLUDE} )

#Catch2 lib
set(CATCH2LIB "${INCLUDE}/catch2/libCatch2.a")


#Wavelet lib
set(WLETLIBSRC "${INCLUDE}/wavelib/wavelet2s.cpp")

#Compile Wavelet lib
#add_library(LibsModule ${WLETLIBSRC})
#target_link_libraries(LibsModule -lfftw3)

#GPU cross correlation
set(GPUccross "${BIN}/crosscorrelation.o")
set(GPUccrossSRC "${SRC}/crosscorrelation.cu")

#Compile GPU cross correlation
add_library(GPUcc ${GPUccrossSRC})
#target_link_libraries(GPUcc)

#Old Main program
set(TARGET "wavebench")
#add_executable(${TARGET} ${EXAMPLES}/${TARGET}.cpp)
#target_link_libraries(${TARGET} LibsModule) # for filesystem: -lstdc++fs)

#Wavelet Demo program
set(TARGET "waveletdemo")
add_executable(${TARGET} ${EXAMPLES}/${TARGET}.cpp)
#target_link_libraries(${TARGET})

#Cuda demo program
set(TARGET "testcuda")
add_executable(${TARGET} ${EXAMPLES}/${TARGET}.cu)
#target_link_libraries(${TARGET})

#Par cross-CORRELATION tests
set(TARGET "parcrosscorrelation")
add_executable(${TARGET} ${EXAMPLES}/${TARGET}.cpp)
#target_link_libraries(${TARGET})

#Unit tests
set(UNITTEST "tests")
add_executable(${UNITTEST} ${TEST}/${UNITTEST}.cpp)
target_link_libraries(${UNITTEST} ${CATCH2LIB})
target_link_libraries(${UNITTEST} GPUcc)


#Run executable
#Run tests
#To do

