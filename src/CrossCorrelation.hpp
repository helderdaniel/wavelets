//
// Created by hdaniel on 02/05/19.
//

/**
 * Note: From Cross-correlation, Convolution can be achieved by reordering
 * 		 the coefficients of the filter by the reverse order (x-axis symmetric).
 *
 * https://en.wikipedia.org/wiki/Cross-correlation
 * https://en.wikipedia.org/wiki/Convolution
 *
 * Convolution can also be done in the operation.
 * Let's say we have an input signal and a filter, to generate the output.
 * Then:
 * (2), the convolution, is about 15% slower than (1), the cross-correlation
 * So having the coefficients of the filter reordered by reverse order
 * previously, will achieve convolution with the same time cost of
 * cross-correlation, since it uses the same operation (1)
 *
 * This is used in WaveletTransform class where Wavelet objects have
 * coefficients by the reverse order (x-axis symmetric).
 *
 *  int ncoefs = filter.size();
 *	for (int i=0; i < output.size(); i+=2) {
 *		T t=0;
 *		for (int j=0; j < ncoefs; ++j)
 *			t += input[i+j] * filter[j];		  //(1) Cross-correlation
 * 			t += input[i+j] * filter[ncoefs-j-1]; //(2) Convolution (x-symmetric filter)
 *		output[pos+i/2]=t;
 *	}
 */

#ifndef __CROSS_CORRELATION_HPP__
#define __CROSS_CORRELATION_HPP__

#include "../src/archtypes.h"

/*
 * Macro to define the convolution loops,
 * which might be object of #pragma omp parallel for
 *
 * needs to be a macro, so that the for loop will be expanded
 * right after #pragma
 * an inline function will NOT do, since #pragma will be
 * before the function call and NOT the loop
 */
#define crossc(input,filter,ncoefs,output,pos) \
	for (int i=0; i < output.size(); i+=2) { \
		T t=0; \
		for (int j=0; j < ncoefs; ++j) \
			t += input[i+j] * filter[j]; \
		output[pos+i/2]=t; \
	}

template<class T, class tag=SEQ>
class CrossCorrelation {
public:
	/**
	 * Computes in output parameter, cross-correlation of input signal
	 * with filter
	 *
	 * @param input:  input signal
	 * @param filter: Filter coefficients
	 * @param output: cross-correlation
	 * @param pos:    start storing output values in output[pos]
	 */
	static void execute(const evector<T> &input, const evector<T> &filter,
			            evector<T> &output, int pos);
}; //class CrossCorrelation SEQ


template<class T>
class CrossCorrelation<T, SEQ> {
public:
	static void execute(const evector<T> &input, const evector<T> &filter,
						evector<T> &output, int pos) {
		crossc(input,filter,filter.size(),output,pos);
	}

}; //class CrossCorrelation SEQ


template<class T>
class CrossCorrelation<T, PAR> {
public:
	static void execute(const evector<T> &input, const evector<T> &filter,
						evector<T> &output, int pos) {
		#pragma omp parallel for //num_threads(1)
		crossc(input,filter,filter.size(),output,pos);
	}

}; //class CrossCorrelation PAR


template<class T>
class CrossCorrelation<T, GPU> {
public:
	static void execute(const evector<T> &input, const evector<T> &filter,
						evector<T> &output, int pos) {
		//NOW is same as SEQ
		//Substitute by GPU here
		//crossc(input,filter,filter.size(),output,pos);
	}

}; //class CrossCorrelation GPU

#endif //__CROSS_CORRELATION_HPP__
