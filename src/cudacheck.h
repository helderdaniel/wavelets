#ifndef __CUDACHECK_H__
#define __CUDACHECK_H__

#include <stdio.h>

/**
 * Check CUDA API calls
 *
 * Macro NDEBUG defined disables checks
 *
 * suggested by:
 * https://github.com/deeperlearning/professional-cuda-c-programming/blob/master/examples/common/common.h
 */

#include <iostream>
using namespace std;

#ifdef NDEBUG
	#define checkError(err,noerr,where,msg)
#else
	#define checkError(err,noerr,where,msg)										\
	if (err != noerr) {                                                      	\
        cerr << where << " error " << err << " '" << msg << "' at line: " << __LINE__ << ":" << __FILE__ << endl; \
		exit(1);                                                                \
        }
#endif


#define checkCUDAlastError()                                                   	\
{                                                                              	\
    const cudaError_t error = cudaGetLastError();								\
    checkError(error,cudaSuccess,"CUDA",cudaGetErrorString(error));				\
}


#define checkCUDA(call)                                                        	\
{                                                                              	\
    const cudaError_t error = call;												\
    checkError(error,cudaSuccess,"CUDA",cudaGetErrorString(error));				\
}

#define checkCUBLAS(call)                                                      	\
{                                                                              	\
    const cublasStatus_t error = call;                                 			\
    checkError(error,CUBLAS_STATUS_SUCCESS,"cuBLAS","");						\
}

#define checkCURAND(call)                                                      	\
{                                                                              	\
    const curandStatus_t error = call;                                         	\
    checkError(error,CURAND_STATUS_SUCCESS,"cuRAND","");						\
}

#define checkCUFFT(call)                                                       	\
{                                                                              	\
    const cufftResult error = call;                                            	\
    checkError(error,CUFFT_SUCCESS,"cuFFT","");									\
}

#define checkCUSPARSE(call)                                                    	\
{                                                                               \
    const cusparseStatus_t error = call;                                        \
    checkError(error,CUSPARSE_STATUS_SUCCESS,"cuSPARSE","");					\
}


#endif //__CUDACHECK_H__