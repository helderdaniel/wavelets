#Compiler Optimize options
#https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html

CC=g++-8
#CC=clang++-7 #-x cuda --cuda-gpu-arch=sm_30 #clang++-8 NOT avail in ubuntu 15 apr 2019
NVCC=nvcc

#INCLUDE     = /home/hdaniel/Dropbox/01-libs/cpp
INCLUDE      = include
CFLAGS       = -Ofast -std=c++17 -I$(INCLUDE) -fopenmp -x cuda -L/usr/local/cuda/lib64 -I/usr/local/cuda/include -lcudart_static -ldl -lrt -lpthread
#CFLAGS       = -Xcompiler -Ofast -Xcompiler -std=c++17 -I$(INCLUDE) -Xcompiler -fopenmp
		        #-g
LDFlags      = #-lfftw3 -lstdc++fs
CUDAFLAGS    = -L/usr/local/cuda/lib64 -lcuda -lcudart
#IF linking with cudart_static adds ~600 000 KBytes to file
#but does NOT need to have cuda runtime installed in computer (check this)

EXAMPLES = examples
SRC      = src
TESTS    = test
BIN		 = bin

#should define mainfile in a parameter when calling make,
#if willing to override this:
#make mainfile=<filename-no-extension>
mainfile = wavebench
testfile = tests
demofile1 = waveletdemo
demofile2 = parcrosscorrelation

CATCH2LIB    = tests-main
CATCH2LIBSRC = $(INCLUDE)/catch2/$(CATCH2LIB).cpp

WLETLIB      = wavelet2s
WLETLIBSRC   = $(INCLUDE)/wavelib/$(WLETLIB).cpp

CROSSCORGPU    = crosscorrelation
CROSSCORGPUSRC = $(SRC)/$(CROSSCORGPU).cu

all: prepare $(BIN)/$(CATCH2LIB).o $(BIN)/$(WLETLIB).o $(BIN)/$(mainfile) $(BIN)/$(testfile) $(BIN)/$(demofile1) $(BIN)/$(demofile2)

prepare:
	-mkdir bin

$(BIN)/$(mainfile): $(BIN)/$(CATCH2LIB).o $(BIN)/$(WLETLIB).o $(EXAMPLES)/$(mainfile).cpp $(SRC)/evector.hpp $(SRC)/wavelet.hpp $(SRC)/wavelettransform.hpp
	#$(CC) $(EXAMPLES)/$(mainfile).cpp $(BIN)/wavelet2s.o $(CFLAGS) -o $(BIN)/$(mainfile) $(LDFlags) -lfftw3

$(BIN)/$(testfile): $(BIN)/$(CROSSCORGPU).o $(BIN)/$(CATCH2LIB).o $(BIN)/$(WLETLIB).o $(TESTS)/$(testfile).cpp $(SRC)/evector.hpp $(SRC)/wavelet.hpp $(SRC)/wavelettransform.hpp
	$(CC) $(TESTS)/$(testfile).cpp $(BIN)/$(CATCH2LIB).o $(BIN)/crosscorrelation.o $(CFLAGS) -o $(BIN)/$(testfile) $(LDFlags) $(CUDAFLAGS)

$(BIN)/$(demofile1): $(EXAMPLES)/$(demofile1).cpp $(SRC)/evector.hpp $(SRC)/wavelet.hpp $(SRC)/wavelettransform.hpp
	$(CC) $(EXAMPLES)/$(demofile1).cpp $(CFLAGS) -o $(BIN)/$(demofile1) $(LDFlags)

$(BIN)/$(demofile2): $(EXAMPLES)/$(demofile2).cpp $(SRC)/evector.hpp $(SRC)/wavelet.hpp $(SRC)/wavelettransform.hpp
	$(CC) $(EXAMPLES)/$(demofile2).cpp $(CFLAGS) -o $(BIN)/$(demofile2) $(LDFlags)

$(BIN)/$(CATCH2LIB).o:
	$(CC) $(CATCH2LIBSRC) $(CFLAGS) $(LDFlags) -c -o $(BIN)/$(CATCH2LIB).o

$(BIN)/$(WLETLIB).o:
	#$(CC) $(WLETLIBSRC) $(CFLAGS) $(LDFlags) -c -o $(BIN)/$(WLETLIB).o

$(BIN)/$(CROSSCORGPU).o: $(CROSSCORGPUSRC)
	$(NVCC) $(CROSSCORGPUSRC) -c -o $(BIN)/$(CROSSCORGPU).o

run:
	cd $(BIN); sudo ionice -c 2 -n 0 nice -n -20 ./$(mainfile)

#https://github.com/catchorg/Catch2/blob/master/docs/command-line.md#specifying-which-tests-to-run
#make unittest tests=~[benchmarks]   #NOT tag
#make unittest tests=Benchmar*       "Starting name with
unittest:
	cd $(BIN); sudo ionice -c 2 -n 0 nice -n -20 ./$(testfile) $(tests)

clean:
	-rm -f $(BIN)/*
	-rm $(BIN)/out1.txt
	-rm $(BIN)/out2.txt
