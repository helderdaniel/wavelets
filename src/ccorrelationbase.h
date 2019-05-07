//
//hdaniel@ualg.pt may 2019
//

#ifndef __CROSSCORRELATIONBASE_H__
#define __CROSSCORRELATIONBASE_H__

#define DOWNSAMPLE 2

/*
 * Macro to define the convolution loops,
 * which might be object of #pragma omp parallel for
 * or CUDA kernel
 *
 * needs to be a macro, so that the for loop will be expanded
 * right after #pragma
 * an inline function will NOT do, since #pragma will be
 * before the function call and NOT the loop
 */

//In just one macro for CPU (SEQ, PAR) and GPU
#define CROSSC(T,input,filter,ncoefs,output,start,end,pos) \
	for (int i = start; i < end; i += DOWNSAMPLE) { \
		T t=0; \
		for (int j=0; j < ncoefs; ++j) \
			t += input[i+j] * filter[j]; \
		output[pos + i / DOWNSAMPLE] = t; \
	}

/*
//In two macros to avoid the loop for GPU implementation
//This one is just for GPU
#define CORRELATION(T,input,filter,ncoefs,output,pos,i) \
 		T t=0; \
		for (int j=0; j < ncoefs; ++j) \
			t += input[i+j] * filter[j]; \
		output[pos + i / DOWNSAMPLE] = t;

//This one can be used by CPU(SEQ and PAR) and GPU
#define CROSSC(T,input,filter,ncoefs,output,start,end,pos) \
	for (int i = start; i < end; i += DOWNSAMPLE) { \
    	CORRELATION(T,input,filter,ncoefs,output,pos,i); \
    }
*/

#endif //__CROSSCORRELATIONBASE_H__