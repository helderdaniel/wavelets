/**
 * Wavelet base implementation
 * hdaniel@ualg.pt apr 2019
 */
#ifndef INC_04_WAVELIB_BENCH_WAVELET_HPP
#define INC_04_WAVELIB_BENCH_WAVELET_HPP

#include "../src/evector.hpp"
using namespace std;

/**
 * Abstract Wavelet base class
 */

/**
 * Wavelet coefficients for low and high pass filter holder
 *
 * @tparam T type of each coefficient
 */
template <typename T>
class Wavelet {
evector<T> lopc, hipc;
int _size;
string _name;

public:
    Wavelet(string name, evector<T> lc, evector<T> hc) {
        lopc = lc;
        hipc = hc;
        _size = lc.size();
        _name = name;
    }

    /**
     * returns low pass filter (Approximation) coefficients
     */
    const evector<T>& lopf() const  { return lopc; }

    /**
     * return high pass filter (Detail) coefficients
     */
    const evector<T>& hipf() const { return hipc; }

    /**
     * returns the number of coefficients in low and high pass filters
     */
    const int size() const { return _size; }

	/**
     * returns the name of the wavelet
 	 */
	const string& name() const { return _name; }

	/**
     * returns a string representing the Wavelet in the form:
     * name
     * lo=lopc()
     * hi=hipc()
 	 */
	string toString() {
		stringstream os;
		os << name() << endl;
		os << "lo: " << lopf() << endl;
		os << "hi: " << hipf();
		return os.str();
	}
};


/**
 * Wavelets factory for: Haar (haar1), Daubechies (db1 (haar1), db2 and db7), ...
 *
 * @tparam T type of each coefficient
 */
template <class T>
class WaveletFactory {
public:
    //haar
	static const Wavelet<T> haar1() {
        return Wavelet<T>("haar1",
                evector<T>{
                    0.7071067811865476,
                    0.7071067811865476
                },
                evector<T>{
                    0.7071067811865476,
                    -0.7071067811865476
                });
    }


    //Daubechies
    static const Wavelet<T> db1() {
        Wavelet<T> h = haar1();
        return Wavelet<T>("db1", h.lopf(), h.hipf());
    }

	static const Wavelet<T> db2() {
		return Wavelet<T>("db2",
				evector<T>{
						0.48296291314469025,
						0.83651630373746899,
						0.22414386804185735,
						-0.12940952255092145
				},
				evector<T>{
						-0.12940952255092145,
						-0.22414386804185735,
						0.83651630373746899,
						-0.48296291314469025
				});
    }

	static const Wavelet<T> db7() {
        return Wavelet<T>("db7",
        		evector<T> {
					0.077852054085062364,
					0.39653931948230575,
					0.72913209084655506,
					0.4697822874053586,
					-0.14390600392910627,
					-0.22403618499416572,
					0.071309219267050042,
					0.080612609151065898,
					-0.038029936935034633,
					-0.01657454163101562,
					0.012550998556013784,
					0.00042957797300470274,
					-0.0018016407039998328,
					0.00035371380000103988
					},
				evector<T> {
					0.00035371380000103988,
					0.0018016407039998328,
					0.00042957797300470274,
					-0.01255099855601378,
					-0.01657454163101562,
					0.038029936935034633,
					0.080612609151065898,
					-0.071309219267050042,
					-0.22403618499416572,
					0.14390600392910627,
					0.4697822874053586,
					-0.72913209084655506,
					0.39653931948230575,
					-0.077852054085062364
					} );
    }
};


#endif