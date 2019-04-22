cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_C_COMPILER /usr/bin/gcc-7)
#set(CMAKE_C_COMPILER /usr/bin/clang-7)
set(CMAKE_CXX_COMPILER /usr/bin/g++-8)
#set(CMAKE_CXX_COMPILER /usr/bin/clang++-7)  #clang++-8 NOT avail in ubuntu 15 apr 2019

#set(GCC_EXTRA_COMPILE_FLAGS "-Ofast")
set(GCC_EXTRA_COMPILE_FLAGS "-O0 -g")
set(GCC_EXTRA_LINK_FLAGS    "")

#reset existing
set(CMAKE_CXX_FLAGS  ${GCC_EXTRA_COMPILE_FLAGS})
set(CMAKE_EXE_LINKER_FLAGS  ${GCC_EXTRA_LINK_FLAGS})

#add to existing
#set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_EXTRA_COMPILE_FLAGS}")
#set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_EXTRA_LINK_FLAGS}")

set(SRC "src")
set(EXAMPLES "examples")
set(TEST "test")

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ..)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ..)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ..)

#hdlib folders
set(INCLUDE "/home/hdaniel/Dropbox/01-libs/cpp")
include_directories(. ${INCLUDE} )

#catch2 lib
set(CATCH2LIB "${INCLUDE}/catch2/tests-main.o")

#Wavelet lib
set(WLETLIBSRC "../00-wavelibsrc/wavelet2s.cpp")

#Project name
project(04_wavelib_bench)

#Compile Wavelet lib
add_library(LibsModule ${WLETLIBSRC})
target_link_libraries(LibsModule -lfftw3)

#Main program
set(TARGET "wavebench")
add_executable(${TARGET} ${EXAMPLES}/${TARGET}.cpp)
target_link_libraries(${TARGET} LibsModule -lstdc++fs)

#Unit tests
set(UNITTEST "tests")
add_executable(${UNITTEST} ${TEST}/${UNITTEST}.cpp)
target_link_libraries(${UNITTEST} ${CATCH2LIB})


#Run executable
#Run tests
#To do

