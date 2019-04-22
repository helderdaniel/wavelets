#Compiler Optimize options
#https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html

CC=g++-8
#CC=clang++-7 #clang++-8 NOT avail in ubuntu 15 apr 2019

INCLUDE   = /home/hdaniel/Dropbox/01-libs/cpp
CATCH2LIB = $(INCLUDE)/catch2/tests-main.o
CFLAGS    = -Ofast -std=c++17 -I$(INCLUDE)
		    #-g
LDFlags   = -lfftw3 -lstdc++fs

EXAMPLES = examples
SRC      = src
TESTS    = test

#should define mainfile in a parameter when calling make,
#if willing to override this:
#make mainfile=<filename-no-extension>
mainfile = wavebench
testfile = tests

WLETLIB=wavelet2s
WLETLIBSRC=../00-wavelibsrc/$(WLETLIB).cpp


all: $(WLETLIB).o $(mainfile) $(testfile)

$(mainfile): $(WLETLIB).o $(EXAMPLES)/$(mainfile).cpp $(SRC)/evector.hpp $(SRC)/wavelet.hpp $(SRC)/wavelettransform.hpp
	$(CC) $(EXAMPLES)/$(mainfile).cpp wavelet2s.o $(CFLAGS) -o $(mainfile) $(LDFlags)

$(testfile): $(WLETLIB).o $(TESTS)/$(testfile).cpp $(SRC)/evector.hpp $(SRC)/wavelet.hpp $(SRC)/wavelettransform.hpp
	$(CC) $(TESTS)/$(testfile).cpp wavelet2s.o $(CFLAGS) $(CATCH2LIB) -o $(testfile) $(LDFlags)

$(WLETLIB).o:
	$(CC) $(WLETLIBSRC) $(CFLAGS) $(LDFlags) -c


run:
	sudo ionice -c 2 -n 0 nice -n -20 ./$(mainfile)

#https://github.com/catchorg/Catch2/blob/master/docs/command-line.md#specifying-which-tests-to-run
#make unittest tests=~[benchmarks]   #NOT tag
#make unittest tests=Benchmar*       "Starting name with
unittest:
	./$(testfile) $(tests)

clean:
	-rm *.o
	-rm *.a
	-rm $(mainfile)
	-rm $(testfile)
	-rm out1.txt
	-rm out2.txt