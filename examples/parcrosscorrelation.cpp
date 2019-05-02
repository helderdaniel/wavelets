/*
 * http://www.modernescpp.com/index.php/multithreading-in-c-17-and-c-20
 * https://hackernoon.com/learn-c-multi-threading-in-5-minutes-8b881c92941f
 * https://eli.thegreenplace.net/2016/c11-threads-affinity-and-hyperthreading/
 *
 * https://bisqwit.iki.fi/story/howto/openmp/#SupportInDifferentCompilers
 */

#include <iostream>
#include <algorithm>
#include <thread>
#include <future>
#include <omp.h>
#include <bits/refwrap.h>
#include <stopwatch/stopwatch.hpp>
#include "../src/wavelet.hpp"
#include "../src/wavelettransform.hpp"
#include "../src/evector.hpp"

#define DTYPE double
#define NUMTHREADS 8

using namespace std;

template <typename T>
struct result {
	T data;
	int cpu;
};

//Call crossc macro for thread launch returning
template <typename T>
result<int> convolution(const evector<T> &input, const evector<T> &filter,
						evector<T> &output, int start, int end, int pos, int par=0) {

	#pragma omp parallel for if(par) //num_threads(4)
	crossc(input,filter,filter.size(),output, start, end, pos);

	result<int> r{ 0, sched_getcpu() };
	return r;
}

//With -Ofast, for this size, implementation with std::async is slower than with OPENMP. why?
//#define SIZE 200000 //000
//For this size is the same
#define SIZE 200000000

int main() {
	auto db7 = WaveletFactory<DTYPE>::db7();
	DTYPE convfunc, convpar = 0;
	int cpu[NUMTHREADS];
	future<result<int>> t[NUMTHREADS];
	//evector<DTYPE> input = { 32, 10, 20, 38, 37, 28, 38, 34, 18, 24, 18, 9, 23, 24, 28, 34 };
	evector<DTYPE> input(SIZE);
	input.symmExt(WaveletTransform::extBeforeSize(db7.size()),
				  WaveletTransform::extAfterSize(db7.size(), input.size()));
	auto output = evector<DTYPE>(input.size()+50);
	StopWatch swfunc, swopenmp, swthread;


	unsigned int c = std::thread::hardware_concurrency();
	cout << "number of cores: " << c << endl;

#pragma omp parallel
	{
		c = omp_get_num_threads(); //need to be inside of #pragma or returns 0
	}
	cout << "number of cores: " << c << endl;

	//this take some seconds for large vectors
	cout << "Generating random vector" << endl;
	srand(time(NULL));
	//generate(input.begin(), input.end(), []() { return rand() % 100; });

	swfunc.reset();
	auto r = convolution<DTYPE>(ref(input), ref(db7.lopf()),
								ref(output), 0, output.size(), 0);
	convfunc = r.data;
	swfunc.lap();

	swopenmp.reset();
	r = convolution<DTYPE>(ref(input), ref(db7.lopf()),
						   ref(output), 0, output.size(), 0, 1);
	convpar = r.data;
	swopenmp.lap();

	int parlen = ceil(((float)output.size()) / NUMTHREADS);
	if (parlen%2==1) ++parlen;

	swthread.reset();
	///Need to launch every thread at the same time first...
	for (int i = 0; i < NUMTHREADS; ++i) {
		int limit = parlen * (i + 1);
		if (limit > output.size()) limit = output.size();
		t[i] = async(std::launch::async, convolution<DTYPE>, ref(input),
					 ref(db7.lopf()), ref(output), parlen * i, limit, 0, 0);
	}
	///and then wait for all threads to end
	for (int i = 0; i < NUMTHREADS; ++i) {
		auto r = t[i].get();
		convpar += r.data;
		cpu[i] = r.cpu;
	}

	swthread.lap();

	cout << fixed << "conv func: " << convfunc << endl;
	cout << "conv par:  " << convpar << endl;
	cout << endl;
	cout << "Speedup:      " << swfunc.watch()/swopenmp.watch() << endl;
	cout << "time func:    " << swfunc.watch() << endl;
	cout << "time openmp:  " << swopenmp.watch() << endl;
	cout << "time threads: " << swthread.watch() << endl;
	cout << "cpus: ";
	for (int i = 0; i< NUMTHREADS; ++i)
		cout << cpu[i] << " ";
	cout << endl;

	return 0;
}
