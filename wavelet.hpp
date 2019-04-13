/**
 * Wavelet base implementation
 * hdaniel@ualg.pt apr 2019
 */
#ifndef __WAVELET_HPP__
#define __WAVELET_HPP__

/**
 * Abstract Wavelet base class
 */
template<class T>
class Wavelet {
T* lowpc;
T* highpc;

public:
    Wavelet(T* lc, int lsize, T* hc, int hsize) {
        
    }

    //Return low (Aproximation) coeficients
    virtual T* lowpf() { return } = 0;
    //Return high (Detail) coeficients
    virtual T* highpf() = 0;
};

template<class T>
class db7 : Wavelet<T> {

public:
    //Return low (Aproximation) coeficients
    virtual T* lowpf() { re}
    //Return high (Detail) coeficients
    virtual T* highpf() = 0;
};

#endif