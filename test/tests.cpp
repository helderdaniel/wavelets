//
// Created by hdaniel on 13/04/19.
//
#include "catch2/catch.hpp"
#include <sstream>
#include <string>
#include <iostream>
#include <stopwatch/stopwatch.h>
#include "../src/evector.hpp"
#include "../src/wavelet.hpp"
#include "../src/wavelettransform.hpp"

#define DTYPE float
#define decimalDigits 3

using namespace std;


TEST_CASE( "Vector operations", "[evector]" ) {
	evector<DTYPE> v0 = {1.1, 2.2, 3.3, 4.4, 5.5};
	string v0str = "[ 1.1 2.2 3.3 4.4 5.5 ]";

    SECTION("Stream evector") {
        stringstream out;
        out << v0;
        REQUIRE(out.str() == v0str);
    }

    SECTION("Vector to string") {
        stringstream out;
        out << v0;
        REQUIRE(v0.toString() == v0str);
    }

    SECTION("Symmetric extension") {
        evector<DTYPE> v1 = v0; //copies evector (needed since symmExt works in place)
        v1.symmExt(3);
        REQUIRE(v1.toString() == "[ 3.3 2.2 1.1 1.1 2.2 3.3 4.4 5.5 5.5 4.4 3.3 ]");
        string r = string("[")+'\n'+"3.3"+'\n'+"2.2";
        REQUIRE(v1.toString('\n') == "[\n3.3\n2.2\n1.1\n1.1\n2.2\n3.3\n4.4\n5.5\n5.5\n4.4\n3.3\n]");
    }
}


TEST_CASE( "Wavelets", "[Wavelet]" ) {
	Wavelet<DTYPE> w0 = WaveletFactory<DTYPE>::haar1();
	Wavelet<DTYPE> w1 = WaveletFactory<DTYPE>::db1();
	Wavelet<DTYPE> w2 = WaveletFactory<DTYPE>::db2();

	REQUIRE(w0.name() == "haar1");
	REQUIRE(w0.size() == 2);
	REQUIRE(w1.name() == "db1");
	REQUIRE(w1.size() == 2);
	REQUIRE(w2.name() == "db2");
	REQUIRE(w2.size() == 4);
	REQUIRE(w1.toString() == w1.name()+"\nlo: [ 0.707107 0.707107 ]\nhi: [ 0.707107 -0.707107 ]");
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
evector<DTYPE> signal0 = { 32, 10, 20, 38, 37, 28, 38, 34, 18, 24, 18, 9, 23, 24, 28, 34 };
const int signalSize = 20000;
evector<DTYPE> signal1(signalSize);


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
		testTransform (WaveletFactory<DTYPE>::haar1(), signal0,
		"[ 29.698 41.012 45.962 50.912 29.698 19.092 33.234 43.841 15.556 -12.728 6.364 2.828 -4.243 6.364 -0.707 -4.243 ]"
		);
	}
	SECTION("DWTdb1") {
		testTransform (WaveletFactory<DTYPE>::db1(), signal0,
		"[ 29.698 41.012 45.962 50.912 29.698 19.092 33.234 43.841 15.556 -12.728 6.364 2.828 -4.243 6.364 -0.707 -4.243 ]"
		);
	}
	SECTION("DWTdb2") {
		testTransform(WaveletFactory<DTYPE>::db2(), signal0,
		"[ 37.477 23.385 46.117 45.410 47.723 31.640 18.271 33.061 45.962 13.472 -8.005 6.322 4.303 -9.072 3.002 3.302 -1.354 3.674 ]"
  		);
	}
	SECTION("DWTdb7") {
		testTransform (WaveletFactory<DTYPE>::db7(), signal0,
		"[ 23.675 43.124 47.685 47.464 29.454 33.200 31.859 46.317 50.747 38.583 22.866 24.840 39.716 46.287 -12.039 17.237 -13.266 13.135 -7.525 -4.178 9.218 -0.833 -0.956 1.871 -6.607 -1.543 9.681 -6.645 ]"
		);
	}
}



TEST_CASE( "Benchmark", "[benchmarks]" ) {
	StopWatch sw;
	auto wvlt = WaveletFactory<DTYPE>::db7();
	const int experiments = 100;
	const int b0runs = 10000;
	const int b1runs = 1;

	//Randomize signal1
	generate(signal1.begin(), signal1.end(), []() { return rand() % 100;} );

	//Nested function must be lambda
	auto benchTransform = []( Wavelet<DTYPE> wvlt,
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


	/**
 	 * Benchmark0 runs a db7 wavelet over a 16 points signal 10000 times
 	 * It does this 100 experiments and presents the evector of time measures and average
     */
	SECTION("Benchmark 0") {
		auto t = benchTransform (wvlt, signal0, b0runs, experiments, sw);
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
		auto t = benchTransform (wvlt, signal1, b1runs, experiments, sw);
		cout << "bench1: " << endl;
		//Print evector of times as a column
		//cout << t.toString('\n') << endl;
		double avg = t.avg();
		cout << avg << endl;
		REQUIRE(avg < 0.005);
	}

}
