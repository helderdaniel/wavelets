cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_C_COMPILER /usr/bin/gcc)
#set(CMAKE_C_COMPILER /usr/bin/clang-7)
#set(CMAKE_CXX_COMPILER /usr/bin/g++-7)
set(CMAKE_CXX_COMPILER /usr/bin/clang++-7)

set(GCC_COVERAGE_COMPILE_FLAGS "-fprofile-arcs -ftest-coverage")
set(GCC_COVERAGE_LINK_FLAGS    "-lgcov")

set(SRC "src")
set(TEST "test")

#hdlib folders
set(INCLUDE "/home/hdaniel/Dropbox/01-libs/cpp")
include_directories(. ${INCLUDE})

#catch2 lib
set(CATCH2LIB "${INCLUDE}/catch2/tests-main.o")

#Project name
project(04_wavelib_bench)

#Main program
set(TARGET "wavebench")
add_library(LibsModule ../00-wavelibsrc/wavelet2s.cpp)
target_link_libraries(LibsModule -lfftw3)

add_executable(${TARGET} ${SRC}/${TARGET}.cpp src/utils.h)
target_link_libraries(${TARGET} LibsModule)

#Unit tests
set(UNITTEST "tests")
add_executable(${UNITTEST} ${TEST}/${UNITTEST}.cpp)
target_link_libraries(${UNITTEST} ${CATCH2LIB})
