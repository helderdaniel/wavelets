//
// Created by hdaniel on 16/04/19.
//

#ifndef INC_04_WAVELETS_WAVELETTRANSFORM_HPP
#define INC_04_WAVELETS_WAVELETTRANSFORM_HPP

#include "wavelet.hpp"
#include "../src/evector.hpp"

class WaveletTransform {

	/**
	 * Computes in output parameter, partial wavelet transform
	 * (low or high filter)
	 *
	 * @param wcoefs: Wavelet coefficients (low or high)
	 * @param input:  signal
	 * @param output: wavelet transform for low or high pass filter
	 * @param pos:    start storing output values in output[pos]
	 * @return position in output of last value stored
	 */
	template <typename T>
	static int dwtp(const evector<T> &wcoefs, const evector<T> &input, evector<T> &output, int pos) {
		const int ncoefs = wcoefs.size();
		int n = pos;

		for (int i=0; i<output.size(); i+=2) {
			double t=0;
			for (int j=0; j<ncoefs; ++j)
				t += input[i+j] * wcoefs[j];
			output[n++]=t;
		}
		return n;
	}

public:

	/**
	 * Returns the number of output values for low pass or high pass filter.
	 * Assumes common case: low and high pass filter have the same number of coefficients
	 *
	 * @param inputSize:  size of input signal
	 * @param wletnCoefs: number of coefficients (low or high)
	 * @return length of transform output vector
	 */
	static int outputSize(int inputSize, int wletnCoefs) {
		int n = ceil((inputSize+wletnCoefs)/2.0)-1; //divide by 2.0 to avoid integer division (which rounds and NOT ceils)
		return n;
	}

	/**
	 * Returns the number of samples to add BEFORE the signal
	 * to compute the wavelet transform on the beginning of the
	 * signal border
	 * @param wletnCoefs: number of coefficients (low or high)
	 * @return number of samples to insert BEFORE
	 */
	static int extBeforeSize(int wletnCoefs) {
		int n = wletnCoefs-2;
		return n;
	}

	/**
	 * Returns the number of samples to add AFTER the signal
	 * to compute the wavelet transform on the beginning of the
	 * signal border
	 * @param wletnCoefs: number of coefficients (low or high)
	 * @return number of samples to insert AFTER
	 */
	static int extAfterSize(int wletnCoefs, int signalSize) {
		int n = wletnCoefs-2;
		n += signalSize % 2;  //if signal size odd then extend one more
		return n;
	}

	/**
	 * Discrete wavelet transform
	 *
	 * @param w:	  Wavelet
	 * @param input:  signal
	 * @param output: vector with result of transform
	 */
	template <typename T>
	static void dwt(const Wavelet<T> &w, const evector<T> &input, evector<T> &output) {
		const int ncoefs = w.size();

		//extend signal to avoid boundaries distortion
		//preserve original signal, since symmExt() works in place
		auto inputExt = input;
		inputExt.symmExt(extBeforeSize(w.size()),
						 extAfterSize(w.size(), input.size()));

		//redim vector here is a BAD idea
		//output = evector<double>(2*(ceil((input.size()+ncoefs)/2)-1));
		//better to have it dimensioned outside, for the maximum, just once
		//check this!!!

		//dwt low and hugh pass
		int n = dwtp(w.lopf(), inputExt, output, 0);
		dwtp(w.hipf(), inputExt, output, n);
	}

};



#endif //INC_04_WAVELIB_BENCH_WAVELETTRANSFORM_H
