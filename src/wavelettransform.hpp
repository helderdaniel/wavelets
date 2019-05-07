//
// Created by hdaniel on 16/04/19.
//

#ifndef INC_04_WAVELETS_WAVELETTRANSFORM_HPP
#define INC_04_WAVELETS_WAVELETTRANSFORM_HPP

#include <cmath>
#include "wavelet.hpp"
#include "evector.hpp"
#include "CrossCorrelation.hpp"

class WaveletTransform {

public:
	/**
	 * Discrete wavelet transform:
	 *
	 * @param w:	  Wavelet
	 * @param input:  signal
	 * @param output: vector with result of transform
	 */
	template<class T, class tag=SEQ>
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

		//DWT
		//use low and high pass symmetric wavelet coefficients to achieve
		//faster convolution using cross-CORRELATION
		CrossCorrelation<T, tag>::execute(inputExt, w.lopfsym(), output, 0);
		CrossCorrelation<T, tag>::execute(inputExt, w.hipfsym(), output, output.size()/2);
		//Non-symmetric wavelet coefficients gives cross-CORRELATION
		//CrossCorrelation::execute(inputExt, w.lopf(), output, 0);
		//CrossCorrelation::execute(inputExt, w.hipf(), output, output.size()/2);
	}

	/**
	 * Returns the number of output values for low pass or high pass filter.
	 * Assumes common case: low and high pass filter have the same number of coefficients
	 *
	 * @param inputSize:  size of input signal
	 * @param wletnCoefs: number of coefficients (low or high)
	 * @return length of partial transform output vector
	 *
	 * PRE: inputSize >= 1 && wletnCoefs
	 */
	static int outputSize(int inputSize, int wletnCoefs) {
		int n = floor((inputSize+wletnCoefs-1)/2.0); //divide by 2.0 to avoid integer division (which rounds and NOT ceils)
		return n;
	}

	/**
	 * Returns the number of output values for each partial output at a level >= 0
	 * Assumes low and high pass filter have the same number of coefficients
	 *
	 * @param inputSize:  size of input signal
	 * @param wletnCoefs: number of coefficients (low or high)
	 * @return length of partial transform output vector at level
	 *
	 * PRE: inputSize >= 1 && wletnCoefs >=2 && level >= 1
	 *
	 * Note: a sequence general term would be better than an iterative way,
	 * to find the return value
	 */
	static int outputSize(int inputSize, int wletnCoefs, int level) {
		for (int i=1; i<=level; ++i)
			inputSize = outputSize(inputSize, wletnCoefs);
		return inputSize;
	}

	/**
	 * Returns the number of samples to add BEFORE the signal
	 * to compute the wavelet transform on the beginning of the
	 * signal border
	 * @param wletnCoefs: number of coefficients (low or high)
	 * @return number of samples to insert BEFORE
	 *
	 * PRE: wletnCoefs >=2
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
	 *
	 * PRE: wletnCoefs >=2 && signalSize >= 1
	 */
	static int extAfterSize(int wletnCoefs, int signalSize) {
		int n = wletnCoefs-2;
		n += signalSize % 2;  //if signal size odd then extend one more
		return n;
	}

};


#endif //INC_04_WAVELETS_WAVELETTRANSFORM_HPP
