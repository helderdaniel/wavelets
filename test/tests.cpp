//
// Created by hdaniel on 13/04/19.
//
#include <sstream>
#include <string>
#include <iostream>
#include <stopwatch/stopwatch.hpp>
#include <catch2/catch.hpp>
#include "../src/evector.hpp"
#include "../src/wavelet.hpp"
#include "../src/wavelettransform.hpp"
#include "../src/archtypes.h"

#define DTYPE double
#define decimalDigits 3

using namespace std;


/***********
 * VECTORS *
 ***********/
TEST_CASE( "Vector operations", "[evector]" ) {
	evector<DTYPE> v0 = {};
	evector<DTYPE> v1 = {1.1};
	evector<DTYPE> v2 = {1.1, 2.2};
	evector<DTYPE> v5 = {1.1, 2.2, 3.3, 4.4, 5.5};
	string v5str = "[ 1.1 2.2 3.3 4.4 5.5 ]";

    SECTION("Stream evector") {
        stringstream out;
        out << v5;
        REQUIRE(out.str() == v5str);
    }

    SECTION("Vector to string") {
        stringstream out;
        out << v5;
        REQUIRE(v5.toString() == v5str);
    }

    SECTION("Symmetric extension") {
        evector<DTYPE> vt1 = v5; //copies evector (needed since symmExt works in place)
		vt1.symmExt(3, 3);
		REQUIRE(vt1.toString() == "[ 3.3 2.2 1.1 1.1 2.2 3.3 4.4 5.5 5.5 4.4 3.3 ]");
		REQUIRE(vt1.toString('\n') == "[\n3.3\n2.2\n1.1\n1.1\n2.2\n3.3\n4.4\n5.5\n5.5\n4.4\n3.3\n]");

		evector<DTYPE> vt2 = v5; //copies evector (needed since symmExt works in place)
								 //needed new evector. reusing vt1 again will fail
        vt2.symmExt(WaveletTransform::extBeforeSize(4),
        		   WaveletTransform::extAfterSize(4, 1));  //odd
        REQUIRE(vt2.toString() == "[ 2.2 1.1 1.1 2.2 3.3 4.4 5.5 5.5 4.4 3.3 ]");

		evector<DTYPE> vt3 = v5; //copies evector (needed since symmExt works in place)
								 //needed new evector. reusing vt1 again will fail
		vt3.symmExt(WaveletTransform::extBeforeSize(4),
				   WaveletTransform::extAfterSize(4, 2));  //even
		REQUIRE(vt3.toString() == "[ 2.2 1.1 1.1 2.2 3.3 4.4 5.5 5.5 4.4 ]");

		evector<DTYPE> vt4 = v0; //copies evector (needed since symmExt works in place)
								 //needed new evector. reusing vt1 again will fail
		REQUIRE_THROWS_AS(vt4.symmExt(1,1), std::length_error);

		evector<DTYPE> vt5 = v1; //copies evector (needed since symmExt works in place)
								 //needed new evector. reusing vt1 again will fail
		vt5.symmExt(2, 2);
		REQUIRE(vt5.toString() == "[ 1.1 1.1 1.1 1.1 1.1 ]");

		evector<DTYPE> vt6 = v2; //copies evector (needed since symmExt works in place)
								 //needed new evector. reusing vt1 again will fail
		vt6.symmExt(5, 5);
		REQUIRE(vt6.toString() == "[ 1.1 1.1 2.2 2.2 1.1 1.1 2.2 2.2 1.1 1.1 2.2 2.2 ]");
	}
}


/************
 * WAVELETS *
 ************/
TEST_CASE( "Wavelets", "[Wavelet]" ) {
	Wavelet<DTYPE> w0 = WaveletFactory<DTYPE>::haar1();
	Wavelet<DTYPE> w1 = WaveletFactory<DTYPE>::db1();
	Wavelet<DTYPE> w2 = WaveletFactory<DTYPE>::db2();
	Wavelet<DTYPE> w3 = Wavelet<DTYPE>("wvlt", evector<DTYPE>{ 1, 2, 3, 4, 5 });

	REQUIRE(w0.name() == "haar1");
	REQUIRE(w0.size() == 2);
	REQUIRE(w1.name() == "db1");
	REQUIRE(w1.size() == 2);
	REQUIRE(w2.name() == "db2");
	REQUIRE(w2.size() == 4);
	REQUIRE(w1.toString() == w1.name()+"\nlo: [ 0.707107 0.707107 ]\nhi: [ -0.707107 0.707107 ]");
	REQUIRE(w3.toString() == string("wvlt")+"\nlo: [ 1 2 3 4 5 ]\nhi: [ -5 4 -3 2 -1 ]");
}



/**********************
 * WAVELET TRANSFORMS *
 **********************/

/**
 * Test input signal
 */
evector<DTYPE> signal0 = { };
evector<DTYPE> signal1 = { 32 };
evector<DTYPE> signal2 = { 32, 10 };
evector<DTYPE> signal16 = { 32, 10, 20, 38, 37, 28, 38, 34, 18, 24, 18, 9, 23, 24, 28, 34 };
const int signalSize = 20000*1;
evector<DTYPE> signal20k(signalSize);

/**
 * Wavelets
 */
auto haar1    = WaveletFactory<DTYPE>::haar1();
auto db1      = WaveletFactory<DTYPE>::db1();
auto db2   	  = WaveletFactory<DTYPE>::db2();
auto db7      = WaveletFactory<DTYPE>::db7();
auto sym3     = WaveletFactory<DTYPE>::sym3();
auto coif4    = WaveletFactory<DTYPE>::coif4();
auto bior6_8  = WaveletFactory<DTYPE>::bior6_8();
auto rbio6_8  = WaveletFactory<DTYPE>::rbio6_8();

/**
 * Expected Wavelet transforms
 */
/*
const double MAX_ERROR = 1e-6;
const evector<DTYPE> sig16haar1 =
		{ 29.698, 41.012, 45.962, 50.912, 29.698, 19.092, 33.234, 43.841,
		 15.556, -12.728, 6.364, 2.828, -4.243, 6.364, -0.707, -4.243 };
*/
string sig16haar1 =
		"[ 29.698 41.012 45.962 50.912 29.698 19.092 33.234 43.841"
		" 15.556 -12.728 6.364 2.828 -4.243 6.364 -0.707 -4.243 ]";
string sig1db1 =
		"[ 45.255 0.000 ]";
string sig2db1 =
		"[ 29.698 15.556 ]";
string sig16db1 =
		"[ 29.698 41.012 45.962 50.912 29.698 19.092 33.234 43.841"
		" 15.556 -12.728 6.364 2.828 -4.243 6.364 -0.707 -4.243 ]";
string sig1db2= //rounded to zero: 9.537e-07
		"[ 45.255 45.255 0.000 0.000 ]";
string sig2db2=
		"[ 37.477 21.920 13.472 -13.472 ]";
string sig16db2=
		"[ 37.477 23.385 46.117 45.410 47.723 31.640 18.271 33.061"
		" 45.962 13.472 -8.005 6.322 4.303 -9.072 3.002 3.302 -1.354 3.674 ]";
string sig1db7= //rounded to (-)zero
		"[ 45.255 45.255 45.255 45.255 45.255 45.255 45.255"
		" 0.000 0.000 0.000 0.000 0.000 0.000 0.000 ]";
string sig2db7=
		"[ 23.931 35.466 23.931 35.466 23.931 35.466 23.931"
  		" -14.448 14.448 -14.448 14.448 -14.448 14.448 -14.448 ]";
string sig16db7=
		"[ 23.675 43.124 47.685 47.464 29.454 33.200 31.859 46.317 50.747 38.583 22.866 24.840 39.716 46.287"
		" -12.039 17.237 -13.266 13.135 -7.525 -4.178 9.218 -0.833 -0.956 1.871 -6.607 -1.543 9.681 -6.645 ]";
string sig1sym3= //rounded to (-)zero
		"[ 45.255 45.255 45.255 0.000 0.000 0.000 ]";
string sig2sym3=
		"[ 16.611 42.786 16.611 8.409 -8.409 8.409 ]";
string sig16sym3=
		"[ 26.676 42.143 20.607 48.501 47.094 43.892 31.296 19.392 33.384 47.364"
		" 9.475 -2.336 2.346 5.430 -10.531 2.306 6.690 -1.457 1.564 -1.862 ]";
string sig1coif4=  //rounded to (-)zero
		"[ 45.255 45.255 45.255 45.255 45.255 45.255 45.255 45.255 45.255 45.255 45.255 45.255"
		" 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 ]";
string sig2coif4=
		"[ 19.246 40.151 19.246 40.151 19.246 40.151 19.246 40.151 19.246 40.151 19.246 40.151"
		" 11.522 -11.522 11.522 -11.522 11.522 -11.522 11.522 -11.522 11.522 -11.522 11.522 -11.522 ]";
string sig16coif4=
		"[ 35.146 20.499 27.980 44.539 47.528 47.510 24.085 38.336 29.024 51.525"
		" 48.438 34.650 22.881 26.378 41.739 46.641 35.146 20.499 27.980"
		" -5.968 -1.694 3.675 -11.489 15.216 -11.089 11.484 -5.325 -5.945"
		" 8.765 0.403 -1.413 2.522 -6.311 -3.057 10.227 -5.968 -1.694 3.675 ]";
string sig1bior6_8= //rounded to (-)zero
		"[ 45.255 45.255 45.255 45.255 45.255 45.255 45.255 45.255 45.255"
		" 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 ]";
string sig2bior6_8=
		"[ 42.363 17.034 42.363 17.034 42.363 17.034 42.363 17.034 42.363"
		" 9.554 -9.554 9.554 -9.554 9.554 -9.554 9.554 -9.554 9.554 ]";
string sig16bior6_8=
		"[ 44.789 46.589 49.237 22.043 40.146 27.642 52.217 48.658"
		" 34.097 23.189 26.337 41.522 46.867 35.337 19.906 28.321"
		" -5.868 -1.539 3.025 -10.238 13.667 -9.549 10.234 -4.694"
		" -6.052 8.673 0.592 -1.570 2.444 -6.233 -2.843 9.949 ]";
string sig1rbio6_8= //rounded to (-)zero
		"[ 45.255 45.255 45.255 45.255 45.255 45.255 45.255 45.255 45.255"
		" 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 ]";
string sig2rbio6_8=
		"[ 39.253 20.144 39.253 20.144 39.253 20.144 39.253 20.144 39.253"
		" 12.664 -12.664 12.664 -12.664 12.664 -12.664 12.664 -12.664 12.664 ]";
string sig16rbio6_8=
		"[ 44.074 47.903 46.970 24.582 38.081 29.200 51.384"
		" 48.468 34.485 23.078 26.468 41.557 46.763 35.053 20.426 28.409"
		" -5.821 -2.245 4.538 -12.486 16.095 -11.629 11.725 -5.363"
		" -6.105 8.976 0.371 -1.468 2.496 -6.243 -3.152 10.309 ]";

/**
 * Generic execution of Wavelet N times
 *
 * @param wvlt		Wavelet
 * @param signal 	input signal
 * @param output 	evector with output of transform operation
 * @param runs 		how many times to run
 * @param sw 		StopWatch to measure execution time
 */
template<class T, class tag>
void doTransform(Wavelet<T> wvlt,
				 const evector<T>& signal,
				 evector<T>& output,
				 const int runs) {

	for (int i=0; i<runs; ++i)
		WaveletTransform::dwt<T, tag>(wvlt, signal, output);
}

/**
 * Generic test wavelet transform operation
 *
 * @param wvlt		Wavelet
 * @param signal 	input signal
 * @param expected 	string with evector output of transform operation
 */
template<class T, class tag>
void
testTransform (Wavelet<DTYPE> wvlt,
					const evector<DTYPE>& signal,
					const string& expected) {
	StopWatch sw;
	const int osize = WaveletTransform::outputSize(signal.size(), wvlt.size());
	auto output = evector<T>(DOWNSAMPLE * osize);
	doTransform<T, tag>(wvlt, signal, output, 1);
	REQUIRE(output.toString(' ', -1, decimalDigits) == expected);
	//REQUIRE(output - expected < MAX_ERROR);
};
/*
template<class T, class tag>
void testTransform1 (Wavelet<DTYPE> wvlt,
					const evector<DTYPE>& signal,
					const evector<DTYPE>& expected) {
	StopWatch sw;
	const int osize = WaveletTransform::outputSize(signal.size(), wvlt.size());
	auto output = evector<T>(DOWNSAMPLE * osize);
	doTransform<T, tag>(wvlt, signal, output, 1;
	REQUIRE((output - expected) < MAX_ERROR);
};
*/
TEST_CASE( "Wavelet transforms utils", "[transforms]" ) {

	SECTION("Output sizes") {
		REQUIRE(WaveletTransform::extBeforeSize(2) == 0);
		REQUIRE(WaveletTransform::extBeforeSize(13) == 11);
		REQUIRE(WaveletTransform::extBeforeSize(14) == 12);

		REQUIRE(WaveletTransform::extAfterSize(2, 1) == 1);
		REQUIRE(WaveletTransform::extAfterSize(2, 2) == 0);
		REQUIRE(WaveletTransform::extAfterSize(13, 16) == 11);
		REQUIRE(WaveletTransform::extAfterSize(14, 16) == 12);

		REQUIRE(WaveletTransform::outputSize(2, 1) == 1);
		REQUIRE(WaveletTransform::outputSize(3, 1) == 1);
		REQUIRE(WaveletTransform::outputSize(4, 1) == 2);
		REQUIRE(WaveletTransform::outputSize(13, 15) == 13);
		REQUIRE(WaveletTransform::outputSize(14, 15) == 14);
		REQUIRE(WaveletTransform::outputSize(13, 16) == 14);
		REQUIRE(WaveletTransform::outputSize(14, 16) == 14);

		REQUIRE(WaveletTransform::outputSize(2, 1, 1) == 1);
		REQUIRE(WaveletTransform::outputSize(3, 1, 1) == 1);
		REQUIRE(WaveletTransform::outputSize(4, 1, 1) == 2);
		REQUIRE(WaveletTransform::outputSize(15, 13, 4) == 12);
		REQUIRE(WaveletTransform::outputSize(15, 14, 4) == 13);
		REQUIRE(WaveletTransform::outputSize(16, 13, 4) == 12);
		REQUIRE(WaveletTransform::outputSize(16, 14, 4) == 13);
		REQUIRE(WaveletTransform::outputSize(50000, 14, 12) == 25);

	}
}

TEMPLATE_TEST_CASE( "Wavelet transforms", "[transforms]", SEQ, PAR, PARa, GPU) {
	SECTION("DWThaar1") {
		evector<DTYPE> output;
		REQUIRE_THROWS_AS(
				WaveletTransform::dwt<DTYPE>(haar1, signal0, output),
				std::length_error);
		testTransform<DTYPE, TestType>(haar1, signal16, sig16haar1);
		//testTransform1<DTYPE, TestType>(haar1, signal16, sig16haar1);
	}
	SECTION("DWTdb1") {
		testTransform<DTYPE, TestType> (db1, signal1, sig1db1);
		testTransform<DTYPE, TestType> (db1, signal2, sig2db1);
		testTransform<DTYPE, TestType> (db1, signal16,sig16db1);
	}
	SECTION("DWTdb2") {
		testTransform<DTYPE, TestType> (db2, signal1, sig1db2);
		testTransform<DTYPE, TestType> (db2, signal2, sig2db2);
		testTransform<DTYPE, TestType> (db2, signal16,sig16db2);
	}

	SECTION("DWTdb7") {
		testTransform<DTYPE, TestType> (db7, signal1, sig1db7);
		testTransform<DTYPE, TestType> (db7, signal2, sig2db7);
		testTransform<DTYPE, TestType> (db7, signal16,sig16db7);
	}
	SECTION("DWTsym3") {
		testTransform<DTYPE, TestType> (sym3, signal1, sig1sym3);
		testTransform<DTYPE, TestType> (sym3, signal2, sig2sym3);
		testTransform<DTYPE, TestType> (sym3, signal16,sig16sym3);
	}
	SECTION("DWTcoif4") {
		testTransform<DTYPE, TestType> (coif4, signal1, sig1coif4);
		testTransform<DTYPE, TestType> (coif4, signal2, sig2coif4);
		testTransform<DTYPE, TestType> (coif4, signal16,sig16coif4);
	}
	SECTION("DWTbior6.8") {
		testTransform<DTYPE, TestType> (bior6_8, signal1, sig1bior6_8);
		testTransform<DTYPE, TestType> (bior6_8, signal2, sig2bior6_8);
		testTransform<DTYPE, TestType> (bior6_8, signal16,sig16bior6_8);
	}
	SECTION("DWTrbio6.8") {
		testTransform<DTYPE, TestType> (rbio6_8, signal1, sig1rbio6_8);
		testTransform<DTYPE, TestType> (rbio6_8, signal2, sig2rbio6_8);
		testTransform<DTYPE, TestType> (rbio6_8, signal16,sig16rbio6_8);
	}
}


/**************
 * BENCHMARKS *
 **************/
template<class T, class tag>
auto benchTransform (
			evector<Wavelet<T>> wvlts,
			const evector<T>& signal,
			const int& runs,
			const int& experiments,
			StopWatch& sw) {
	evector<double> t(experiments);
	//dim output for maximum len wavelet
	const int osize = WaveletTransform::outputSize(signal.size(), bior6_8.size());
	auto output = evector<T>(DOWNSAMPLE * osize);

	for (int i=0; i<experiments; ++i) {
		//cout << "exp: " << i << endl;
		sw.reset();
		for (auto wvlt : wvlts)
			doTransform<T, tag>(wvlt, signal, output, runs);

		//Is about 15% slower
		//#pragma omp parallel for
		//required old for style by #pragma omp
		/*for (int i=0; i<wvlts.size(); ++i)
			doTransform<T, tag>(wvlts[i], signal, output, runs);
		*/
		sw.lap();
		t[i] = sw.watch();
	}
	return t;
}

template<class T, class tag>
void doBench(evector<Wavelet<T>> wvlts,
			 const evector<T>& signal,
			 const int& runs,
			 const int& experiments,
			 StopWatch& sw,
			 string benchName,
			 double maxTime) {

	auto t = benchTransform<T, tag>(wvlts, signal, runs, experiments, sw);
	double avg = t.avg();
	cout << benchName << ": " << fixed << avg << endl;
	//Print evector of times as a column
	//cout << t.toString('\n') << endl;
	//REQUIRE(avg < maxTime);
}


TEST_CASE( "Wavelet Benchmarks", "[benchmarks]" ) {
	StopWatch sw;
	const int experiments = 100;
	const int b0runs = 10000;
	const int b1runs = 1;

	//Randomize signal20k
	srand(time(NULL));
	generate(signal20k.begin(), signal20k.end(), []() { return rand() % 100; } );

	/**
 	 * Benchmark0 runs a db7 wavelet over a 16 points signal 10000 times
 	 * It does this 100 experiments and presents the evector of time measures and average
     */
	SECTION("Benchmark 0") {
		evector<Wavelet<DTYPE>> wvlts = { db7 };
		doBench<DTYPE, SEQ>(wvlts, signal16, b0runs, experiments, sw, "bench0 (seq)", 0.1);
		//Too slow (input vector have only 16 elements)
		//doBench<DTYPE, PAR>(wvlts, signal16, b0runs, experiments, sw, "bench0 (par)", 0.005);
		//doBench<DTYPE, GPU>(wvlts, signal16, b0runs, experiments, sw, "bench0 (gpu)", 0.005);
	}


	/**
	 * Benchmark1 runs a db7 wavelet over a 20000 points signal 1 time
	 * It does this 100 experiments and presents the evector of time measures and average
	 */
	SECTION("Benchmark 1") {
		evector<Wavelet<DTYPE>> wvlts = { db7 };
		doBench<DTYPE, SEQ>(wvlts, signal20k, b1runs, experiments, sw, "bench1 (seq)", 0.005);
		doBench<DTYPE, PAR>(wvlts, signal20k, b1runs, experiments, sw, "bench1 (par)", 0.005);
		doBench<DTYPE, GPU>(wvlts, signal20k, b1runs, experiments, sw, "bench1 (gpu)", 0.005);
	}

	/**
	 * Benchmark2 runs several Wavelets over a 20000 points signal 1 time
	 * It does this 100 experiments and presents the evector of time measures and average
	 */
	SECTION("Benchmark 2") {
		evector<Wavelet<DTYPE>> wvlts = { haar1, db1, db2, db7, coif4, bior6_8, rbio6_8 };
		doBench<DTYPE, SEQ>(wvlts, signal20k, b1runs, experiments, sw, "bench2 (seq)", 0.005);
		doBench<DTYPE, PAR>(wvlts, signal20k, b1runs, experiments, sw, "bench2 (par)", 0.005);
		doBench<DTYPE, GPU>(wvlts, signal20k, b1runs, experiments, sw, "bench2 (gpu)", 0.005);
	}

}
