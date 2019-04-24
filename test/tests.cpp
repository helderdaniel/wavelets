//
// Created by hdaniel on 13/04/19.
//
#include "catch2/catch.hpp"
#include <sstream>
#include <string>
#include <iostream>
#include <stopwatch/stopwatch.hpp>
#include "../src/evector.hpp"
#include "../src/wavelet.hpp"
#include "../src/wavelettransform.hpp"

#define DTYPE double
#define decimalDigits 3

using namespace std;


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


TEST_CASE( "Wavelets", "[Wavelet]" ) {
	Wavelet<DTYPE> w0 = WaveletFactory<DTYPE>::haar1();
	Wavelet<DTYPE> w1 = WaveletFactory<DTYPE>::db1();
	Wavelet<DTYPE> w2 = WaveletFactory<DTYPE>::db2();
	Wavelet<DTYPE> w3 = Wavelet("wvlt", evector<DTYPE>{ 1, 2, 3, 4, 5 });

	REQUIRE(w0.name() == "haar1");
	REQUIRE(w0.size() == 2);
	REQUIRE(w1.name() == "db1");
	REQUIRE(w1.size() == 2);
	REQUIRE(w2.name() == "db2");
	REQUIRE(w2.size() == 4);
	REQUIRE(w1.toString() == w1.name()+"\nlo: [ 0.707107 0.707107 ]\nhi: [ 0.707107 -0.707107 ]");
	REQUIRE(w3.toString() == string("wvlt")+"\nlo: [ 1 2 3 4 5 ]\nhi: [ 5 -4 3 -2 1 ]");
}


/**
 * Generic test wavelet transform operation
 *
 * @param wvlt		Wavelet
 * @param signal 	input signal
 * @param output 	evector with output of transform operation
 * @param runs 		how many times to run
 * @param sw 		StopWatch to measure execution time
 */
void doTransform(	Wavelet<DTYPE> wvlt,
					const evector<DTYPE>& signal,
					evector<DTYPE>& output,
					const int runs,
					StopWatch &sw ) {
		sw.reset();
		for (int i=0; i<runs; ++i)
			//DWT
			WaveletTransform::dwt<DTYPE>(wvlt, signal, output);
		sw.lap();
}

/**
 * Test input signal
 */
evector<DTYPE> signal0 = { };
evector<DTYPE> signal1 = { 32 };
evector<DTYPE> signal2 = { 32, 10 };
evector<DTYPE> signal16 = { 32, 10, 20, 38, 37, 28, 38, 34, 18, 24, 18, 9, 23, 24, 28, 34 };
const int signalSize = 20000;
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

TEST_CASE( "Wavelet transforms", "[transforms]" ) {
	//Nested function must be lambda
	auto testTransform = []( Wavelet<DTYPE> wvlt,
						const evector<DTYPE>& signal,
						const string& expected) {

		StopWatch sw;
		const int osize = WaveletTransform::outputSize(signal.size(), wvlt.size());
		auto output = evector<DTYPE>(2 * osize);
		doTransform(wvlt, signal, output, 1, sw);
		REQUIRE(output.toString(' ', -1, decimalDigits) == expected);
	};

	SECTION("DWThaar1") {
		evector<DTYPE> output;
		REQUIRE_THROWS_AS(
				WaveletTransform::dwt<DTYPE>(haar1, signal0, output),
				std::length_error);
		testTransform (haar1, signal16,
		"[ 29.698 41.012 45.962 50.912 29.698 19.092 33.234 43.841 "
        "15.556 -12.728 6.364 2.828 -4.243 6.364 -0.707 -4.243 ]"
		);
	}
	SECTION("DWTdb1") {
		testTransform (db1, signal1, "[ 45.255 0.000 ]");
		testTransform (db1, signal2, "[ 29.698 15.556 ]");
		testTransform (db1, signal16,
		"[ 29.698 41.012 45.962 50.912 29.698 19.092 33.234 43.841 "
        "15.556 -12.728 6.364 2.828 -4.243 6.364 -0.707 -4.243 ]"
		);
	}
	SECTION("DWTdb2") {
		testTransform (db2, signal1, "[ 45.255 45.255 0.000 0.000 ]"); //rounded to zero: 9.537e-07 9.537e-07 ]");
		testTransform (db2, signal2, "[ 37.477 21.920 13.472 -13.472 ]");
		testTransform(db2, signal16,
		"[ 37.477 23.385 46.117 45.410 47.723 31.640 18.271 33.061 45.962 13.472 -8.005 6.322 4.303 -9.072 3.002 3.302 -1.354 3.674 ]"
  		);
	}
	SECTION("DWTdb7") {
		testTransform(db7, signal1,
		"[ 45.255 45.255 45.255 45.255 45.255 45.255 45.255 "
        "0.000 0.000 0.000 0.000 0.000 0.000 0.000 ]"); //rounded to (-)zero
		testTransform(db7, signal2,
		"[ 23.931 35.466 23.931 35.466 23.931 35.466 23.931 "
        "-14.448 14.448 -14.448 14.448 -14.448 14.448 -14.448 ]");
		testTransform(db7, signal16,
		"[ 23.675 43.124 47.685 47.464 29.454 33.200 31.859 46.317 50.747 38.583 22.866 24.840 39.716 46.287 "
        "-12.039 17.237 -13.266 13.135 -7.525 -4.178 9.218 -0.833 -0.956 1.871 -6.607 -1.543 9.681 -6.645 ]"
		);
	}
	SECTION("DWTsym3") {
		testTransform (sym3, signal1, "[ 45.255 45.255 45.255 0.000 0.000 0.000 ]"); //rounded to (-)zero
		testTransform (sym3, signal2, "[ 16.611 42.786 16.611 8.409 -8.409 8.409 ]");
		testTransform (sym3, signal16,
	   "[ 26.676 42.143 20.607 48.501 47.094 43.892 31.296 19.392 33.384 47.364 "
	   "9.475 -2.336 2.346 5.430 -10.531 2.306 6.690 -1.457 1.564 -1.862 ]"
		);
	}
	SECTION("DWTcoif4") {
		testTransform (coif4, signal1,
		"[ 45.255 45.255 45.255 45.255 45.255 45.255 45.255 45.255 45.255 45.255 45.255 45.255 "
		"0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 ]"); //rounded to (-)zero
		testTransform (coif4, signal2,
		"[ 19.246 40.151 19.246 40.151 19.246 40.151 19.246 40.151 19.246 40.151 19.246 40.151 "
		"11.522 -11.522 11.522 -11.522 11.522 -11.522 11.522 -11.522 11.522 -11.522 11.522 -11.522 ]");
		testTransform (coif4, signal16,
		"[ 35.146 20.499 27.980 44.539 47.528 47.510 24.085 38.336 29.024 51.525 "
        "48.438 34.650 22.881 26.378 41.739 46.641 35.146 20.499 27.980 "
		"-5.968 -1.694 3.675 -11.489 15.216 -11.089 11.484 -5.325 -5.945 "
        "8.765 0.403 -1.413 2.522 -6.311 -3.057 10.227 -5.968 -1.694 3.675 ]"
		);
	}
	SECTION("DWTbior6.8") {
		testTransform (bior6_8, signal1,
	    "[ 45.255 45.255 45.255 45.255 45.255 45.255 45.255 45.255 45.255 45.255 45.255 45.255 "
	    "0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 ]"); //rounded to (-)zero
		testTransform (bior6_8, signal2,
		"[ 19.246 40.151 19.246 40.151 19.246 40.151 19.246 40.151 19.246 40.151 19.246 40.151 "
		"11.522 -11.522 11.522 -11.522 11.522 -11.522 11.522 -11.522 11.522 -11.522 11.522 -11.522 ]");
		testTransform (bior6_8, signal16,
		"[ 35.146 20.499 27.980 44.539 47.528 47.510 24.085 38.336 29.024 51.525 "
		"48.438 34.650 22.881 26.378 41.739 46.641 35.146 20.499 27.980 "
        "-5.968 -1.694 3.675 -11.489 15.216 -11.089 11.484 -5.325 -5.945 "
		"8.765 0.403 -1.413 2.522 -6.311 -3.057 10.227 -5.968 -1.694 3.675 ]"
		);
	}
	SECTION("DWTrbio6.8") {
		testTransform (rbio6_8, signal1,
					   "[ 45.255 45.255 45.255 45.255 45.255 45.255 45.255 45.255 45.255 "
					   "0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 ]"); //rounded to (-)zero
		testTransform (rbio6_8, signal2,
					   "[ 42.363 17.034 42.363 17.034 42.363 17.034 42.363 17.034 42.363 "
					   "9.554 -9.554 9.554 -9.554 9.554 -9.554 9.554 -9.554 9.554 ]");
		testTransform (rbio6_8, signal16,
					   "[ 44.789 46.589 49.237 22.043 40.146 27.642 52.217 48.658 34.097 23.189 26.337 41.522 46.867 35.337 19.906 28.321 "
					   "-5.868 -1.539 3.025 -10.238 13.667 -9.549 10.234 -4.694 -6.052 8.673 0.592 -1.570 2.444 -6.233 -2.843 9.949 ]"
		);
	}

}

//if nested function, must be lambda
//auto benchTransform = [](
auto benchTransform (
			Wavelet<DTYPE> wvlt,
			const evector<DTYPE>& signal,
			const int& runs,
			const int& experiments,
			StopWatch& sw) {
	evector<double> t(experiments);
	const int osize = WaveletTransform::outputSize(signal.size(), wvlt.size());
	auto output = evector<DTYPE>(2 * osize);

	for (int i=0; i<experiments; ++i) {
		//cout << "exp: " << i << endl;
		doTransform(wvlt, signal, output, runs, sw);
		t[i] = sw.cpuTime();
	}
	return t;
};

TEST_CASE( "Benchmarkdb7", "[benchmarks]" ) {
	StopWatch sw;
	auto wvlt = WaveletFactory<DTYPE>::db7();
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
		auto t = benchTransform (wvlt, signal16, b0runs, experiments, sw);
		cout << "bench0: " << endl;
		//Print evector of times as a column
		//cout << t.toString('\n') << endl;
		double avg = t.avg();
		cout << avg << endl;
		REQUIRE(avg < 0.005);
	}


	/**
	 * Benchmark1 runs a db7 wavelet over a 20000 points signal 1 time
	 * It does this 100 experiments and presents the evector of time measures and average
	 */
	SECTION("Benchmark 1") {
		auto t = benchTransform (wvlt, signal20k, b1runs, experiments, sw);
		cout << "bench1: " << endl;
		//Print evector of times as a column
		//cout << t.toString('\n') << endl;
		double avg = t.avg();
		cout << avg << endl;
		REQUIRE(avg < 0.005);
	}

}
