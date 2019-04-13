#Copmpiler Optimize options
#https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html

CC=g++-7
#CC=clang++-7
LIBS=/home/hdaniel/Dropbox/01-libs/cpp
CATCH2=${LIBS}/catch2/tests-main.o
CFLAGS= -Ofast -std=c++17 -I${LIBS} ${CATCH2}
#-g
LDFlags=-lfftw3

#should define mainfile in a parameter when calling make,
#if willing to override:
#make mainfile=<filename-no-extensin>
mainfile=wavebench

all: 
	$(CC) $(mainfile).cpp ../00-wavelibsrc/wavelet2s.cpp $(CFLAGS) -o $(mainfile) $(LDFlags)

run:
	sudo ionice -c 2 -n 0 nice -n -20 ./$(mainfile)

clean:
	rm $(mainfile)
 
