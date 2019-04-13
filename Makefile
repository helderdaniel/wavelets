#Compiler Optimize options
#https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html

CC=g++-7
#CC=clang++-7

INCLUDE   = /home/hdaniel/Dropbox/01-libs/cpp
CATCH2LIB = $(INCLUDE)/catch2/tests-main.o
CFLAGS    = -Ofast -std=c++17 -I$(INCLUDE)
		    #-g
LDFlags   = -lfftw3

SRC  = src
TESTS = test

#should define mainfile in a parameter when calling make,
#if willing to override this:
#make mainfile=<filename-no-extension>
mainfile = wavebench
testfile = tests

WLETLIB=wavelet2s
WLETLIBSRC=../00-wavelibsrc/$(WLETLIB).cpp


all: $(WLETLIB).o $(mainfile) $(testfile)

$(mainfile): $(WLETLIB).o $(SRC)/$(mainfile).cpp $(SRC)/utils.h
	$(CC) $(SRC)/$(mainfile).cpp wavelet2s.o $(CFLAGS) -o $(mainfile) $(LDFlags)

$(testfile): $(WLETLIB).o $(TESTS)/$(testfile).cpp $(SRC)/utils.h
	$(CC) $(TESTS)/$(testfile).cpp wavelet2s.o $(CFLAGS) $(CATCH2LIB) -o $(testfile) $(LDFlags)

$(WLETLIB).o:
	$(CC) $(WLETLIBSRC) $(CFLAGS) $(LDFlags) -c


run:
	sudo ionice -c 2 -n 0 nice -n -20 ./$(mainfile)

unittest:
	./$(testfile)

clean:
	-rm wavelet2s.o
	-rm $(mainfile)
	-rm $(testfile)

