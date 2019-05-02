/**
 * Simple dwt db7 benchmark and test for wavelib
 * 
 * hdaniel@ualg.pt apr 2019
 */
/*
read matlab
https://github.com/hbristow/cvmatio (does not write)
https://sourceforge.net/projects/matio/

fft:
http://www.fftw.org/

*/

#include <iostream>
#include <fstream>
#include <thread>
#include <numeric>
#include "stopwatch/stopwatch.hpp"
#include "../include/wavelib/wavelet2s.h"
#include "../src/evector.hpp"
#include "../src/wavelet.hpp"
#include "../src/wavelettransform.hpp"

#define DEBUG
#define BENCH1

using namespace std;


evector<double> test();
int main() {
#ifdef DEBUG
	const int times = 1;
#else
	const int times = 100;
#endif
evector<evector<double>> t(2);

	for (int i=0; i<times; i++) {
		cout << i << endl;
		evector<double> r = test();
		t[0].push_back(r[0]);
		t[1].push_back(r[1]);
	}

	cout << endl;
#ifndef DEBUG
	for (int i=0; i<2; ++i)
		//Print vector as a column
		cout << t[i].toString('\n') << endl;
#endif
	for (int i=0; i<2; ++i)
		cout << "avg = " << accumulate(t[i].begin(), t[i].end(), 0.0) / t[i].size() << endl;

	{
		unsigned int c = std::thread::hardware_concurrency();
		std::cout << " number of cores: " << c << endl;;
	}

}

evector<double> test() {
StopWatch sw;

#ifdef BENCH1
	const int runs = 10000;
#else
	const int runs = 1;
#endif


Wavelet wvlt = WaveletFactory<double>::db7();
const int ncoefs = wvlt.size();
#ifdef BENCH1
	evector<double> signal = { 32, 10, 20, 38, 37, 28, 38, 34, 18, 24, 18, 9, 23, 24, 28, 34 };
#else
	const int signalSize = 20000;
	evector<double> signal(signalSize);
	generate(signal.begin(), signal.end(), []() { return rand() % 100;} );
#endif
int osize = WaveletTransform::outputSize(signal.size(), ncoefs);
evector<double> output1(2*osize);
//evector<double> output1(2*(ceil((signal.size()+ncoefs)/2)-1));
evector<double> output2, flag;
evector<int> length;


	//cout << signal.size() << endl;
	sw.reset();
	for (int i = 0; i<runs; ++i) {
		WaveletTransform::dwt<double>(wvlt, signal, output1);
	}
    sw.lap();
    double rt0 = sw.watch();
    cout << sw << endl;

    sw.reset();
    for (int i = 0; i<runs; ++i) {
		output2.clear();
		dwt_sym(signal, 1, wvlt.name(), output2, flag, length);
	}
    sw.lap();
	double rt1 = sw.watch();
    cout << sw << endl;

    cout << output1.size() << endl;
    cout << output2.size() << endl;

    //diff by sum of squared errors (SSE)
    double diff=0;
    for (int i=0; i<output1.size(); ++i)
        diff += pow(output1[i]-output2[i], 2);
    cout << diff/output1.size() << endl;

    #ifdef DEBUG
    	cout << output1 << endl;
		cout << output2 << endl;
    #endif

	//needed to avoid optimizer to eliminate code
    ofstream out1("out1.txt");
	ofstream out2("out2.txt");
    out1 << output1 << endl;
    out2 << output2 << endl;

    evector<double> r = { rt0, rt1 };
    return r;
}