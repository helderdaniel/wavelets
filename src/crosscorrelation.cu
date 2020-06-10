/**
 * cross correlation implemntation in CUDA
 * called from crosscorrelation.hpp
 * linked with main file
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include "ccorrelationbase.h"
#include <evector/evector.hpp>

using namespace std;
using namespace had;

//#define DEBUG
#ifdef DEBUG
#include "wavelet.hpp"
#include "wavelettransform.hpp"
#include <iostream>
using namespace std;
#endif

#define DTYPE double


//GPU kernel function
#define SPREAD 1 //No apparent gain in spreading the process of more than an elemnt by thread
__global__
void cckernel(const DTYPE *input, const DTYPE *filter, int ncoefs, DTYPE *output, int osize, int pos) {
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const int inputIdx = threadId * DOWNSAMPLE * SPREAD; //Dual downsample

    if (inputIdx < osize) {
		CROSSC(DTYPE,input,filter,ncoefs,output,inputIdx,inputIdx+SPREAD*DOWNSAMPLE,pos);
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
	int cols = 256/SPREAD;
	int rows = ceil(((float)input.size())/DOWNSAMPLE/cols); //force floating point
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
//CREATE CUDA class
//on constructor verify cuda availability
//https://github.com/deeperlearning/professional-cuda-c-programming/blob/master/solutions/common/common.h
////also CHECKCUDA from addvector cuda project?

//do warmup
//https://github.com/deeperlearning/professional-cuda-c-programming/blob/master/examples/chapter03/simpleDivergence.cu


#ifdef DEBUG
//to compile test: nvcc crosscorrelation.cu
//
int main() {
auto w = WaveletFactory<DTYPE>::haar1();

evector<DTYPE> input = { 32, 10, 20, 38, 37, 28, 38, 34, 18, 24, 18, 9, 23, 24, 28, 34 };
//evector<DTYPE> input = { 32, 10 };
evector<DTYPE> filter = { 0.7071067811865476, 0.7071067811865476 };
const int osize = WaveletTransform::outputSize(input.size(), w.size());
auto output = evector<DTYPE>(DOWNSAMPLE * osize);

	ccorrelation(input, w.lopf(), output, 0);
	ccorrelation(input, w.hipfsym(), output, output.size()/DOWNSAMPLE);
	cout << input.toString(5) << endl;
	cout << filter.toString(5) << endl;
	cout << output.toString(5) << endl;
	return 0;
}
#endif