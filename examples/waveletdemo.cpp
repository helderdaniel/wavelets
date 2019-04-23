/**
 * Simple Wavelets demo
 * 
 * hdaniel@ualg.pt apr 2019
 *
 * compile with:
 *
 * g++-7 waveletdemo.cpp -o waveletdemo -std=c++17
 *
 */

//include wavelet and extended vector
#include "../src/evector.hpp"
#include "../src/wavelet.hpp"
#include "../src/wavelettransform.hpp"

#include <iostream>
using namespace std;

#define DTYPE double

int main() {
	//define Wavelet here: haar1, db1, db2, db7
	Wavelet<DTYPE> wvlt = WaveletFactory<DTYPE>::db7();

	//input signal
	evector<DTYPE> input = { 32, 10, 20, 38, 37, 28, 38, 34, 18, 24, 18, 9, 23, 24, 28, 34 };

	//Dim output signal vector according to Wavelet coefficients
	const int wvltcoefs = wvlt.size();
	const int osize = WaveletTransform::outputSize(input.size(), wvltcoefs);
	auto output = evector<DTYPE>(2 * osize);

	//Perform transform
	WaveletTransform::dwt<DTYPE>(wvlt, input, output);

	cout << "Input signal:\n" << input << endl << endl;
	cout << "Discrete Wavelet Transform (" + wvlt.name() + "):\n" << output << endl;

	return 0;
}