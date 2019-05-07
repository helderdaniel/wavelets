/**
 * cross correlation implemntation in CUDA
 * called from crosscorrelation.hpp
 * linked with main file
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include "ccorrelationbase.h"
#include "evector.hpp"

//#define DEBUG
#ifdef DEBUG
#include "wavelet.hpp"
#include "wavelettransform.hpp"
#include <iostream>
using namespace std;
#endif

#define DTYPE double

/*
#define DOWNSAMPLE 2
#define CROSSC2(T, input,filter,ncoefs,output,start,end,pos) \
	//for (int i = start; i < end; i += DOWNSAMPLE)  \
	{ \
		T t = 0; \
		const int i = start; \
		for (int j = 0; j < ncoefs; ++j) \
			t += input[i + j] * filter[j]; \
		output[pos + i / DOWNSAMPLE] = t; \
	}
*/

//GPU kernel function
__global__
void cckernel(const DTYPE *input, const DTYPE *filter, int ncoefs, DTYPE *output, int osize, int pos) {
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const int inputIdx = threadId * DOWNSAMPLE; //Dual downsample

    if (threadId < osize) {
		CROSSC(DTYPE,input,filter,ncoefs,output,inputIdx,inputIdx+1,pos);
		//CORRELATION(DTYPE,input,filter,ncoefs,output,pos,inputIdx);
    }
}


//Launch GPU kernel function
void ccorrelation(const evector<DTYPE> &input, const evector<DTYPE> &filter,
                    evector<DTYPE> &output, int pos) {

	// Allocate GPU device
	DTYPE *ginput, *gfilter, *goutput;

	cudaMalloc((void**)&ginput,  sizeof(DTYPE)*input.size());
	cudaMalloc((void**)&gfilter, sizeof(DTYPE)*filter.size());
	cudaMalloc((void**)&goutput, sizeof(DTYPE)*output.size()/DOWNSAMPLE);

	// Transfer data from host to GPU device memory
	cudaMemcpy(ginput,  input.data(),  sizeof(DTYPE)*input.size(),  cudaMemcpyHostToDevice);
	cudaMemcpy(gfilter, filter.data(), sizeof(DTYPE)*filter.size(), cudaMemcpyHostToDevice);

	//launch GPU threads
	int rows = ceil(((float)input.size())/DOWNSAMPLE/256); //force floating point
	int cols = ceil(((float)input.size())/DOWNSAMPLE/rows);
	#ifdef DEBUG
		cout << rows << "x" << cols << endl;
	#endif
	cckernel<<<rows,cols>>>(ginput,gfilter,filter.size(),goutput,output.size(),0);

	// Transfer data from GPU device to host memory
	cudaMemcpy(&(output.data()[pos]), goutput, sizeof(DTYPE)*output.size()/DOWNSAMPLE, cudaMemcpyDeviceToHost);

	// Deallocate device memory
	cudaFree(ginput);
	cudaFree(goutput);
	cudaFree(goutput);
}


#ifdef DEBUG
//to compile test: nvcc crosscorrelation.cu
//
int main() {
auto w = WaveletFactory<DTYPE>::haar1();

evector<DTYPE> input = { 32, 10, 20, 38, 37, 28, 38, 34, 18, 24, 18, 9, 23, 24, 28, 34 };
//evector<DTYPE> input = { 32, 10 };
evector<DTYPE> filter = { 0.7071067811865476, 0.7071067811865476 };
const int osize = WaveletTransform::outputSize(input.size(), w.size());
auto output = evector<DTYPE>(2 * osize);

	ccorrelation(input, w.lopf(), output, 0);
	ccorrelation(input, w.hipfsym(), output, output.size()/2);
	cout << output.toString(5) << endl;
	return 0;
}
#endif