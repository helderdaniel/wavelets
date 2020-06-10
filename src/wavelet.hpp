/**
 * Wavelet base implementation
 * hdaniel@ualg.pt apr 2019
 */
#ifndef INC_04_WAVELETS_WAVELET_HPP
#define INC_04_WAVELETS_WAVELET_HPP

#include <evector/evector.hpp>
using namespace std;
using namespace had;

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
//wavelet coefficients low and high pass filters
evector<T> lopc, hipc;
//symmetric wavelet coefficients for fast convolution
evector<T> lopcsym, hipcsym;
int _size;
string _name;

	/**
	 * base object initialization
	 *
	 * @param name: Name of wavelet
	 * @param lc:   low pass filter coefficients
	 * @param hc: 	high pass filter coefficients
	 */
	void WaveletInit(string name, evector<T> lc, evector<T> hc) {
		lopc = lc;
		lopcsym = evector<T>(lc.rbegin(), lc.rend());
		hipc = hc;
		hipcsym = evector<T>(hc.rbegin(), hc.rend());
		_size = lc.size();
		_name = name;
	}

public:
	/**
	 * @param name: Name of wavelet
	 * @param lc:   low pass filter coefficients
	 * 				high pass filter coefficients computed:
	 * 					invert order of low pass filter coefs
	 * 					if odd
	 * 						multiply odd indexes by -1
	 * 					else
	 * 						multiply even indexes by -1
	 * @param odd:
	 */
	Wavelet(string name, evector<T> lc, bool odd=false) {

		//reverse low coefs
		//redone in WaveletInit() to get high pass coefs
		//could be optimized, but wavelet init can be done only once
		auto hc = evector<T>(lc.rbegin(), lc.rend());

		//multiply even or odd coefs by -1
		const int startidx = (odd) ? 1 : 0;
		for(int i = startidx; i < hc.size(); i+=2)
			hc[i] *= -1;

		WaveletInit(name, lc, hc);
	}

	/**
	 * @param name: Name of wavelet
	 * @param lc:   low pass filter coefficients
	 * @param hc: 	high pass filter coefficients
	 */
    Wavelet(string name, evector<T> lc, evector<T> hc) {
    	WaveletInit(name, lc, hc);
    }

    /**
     * lopf()    returns low pass filter (Approximation) coefficients
     * lopfsym() returns low pass filter symmetric coefficients
     * 		     useful for fast convolution in FWT algorithm
     *
     */
    const evector<T>& lopf() const  { return lopc; }
	const evector<T>& lopfsym() const  { return lopcsym; }

    /**
     * hipf()    return high pass filter (Detail) coefficients
     * hipfsym() returns high pass filter symmetric coefficients
     * 		     useful for fast convolution in FWT algorithm
     */
    const evector<T>& hipf() const { return hipc; }
	const evector<T>& hipfsym() const  { return hipcsym; }

    /**
     * returns the number of coefficients in low and high pass filters
     */
    int size() const { return _size; }

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
	friend string to_string(const Wavelet& w) {
		stringstream os;
		os << w.name() << endl;
		os << "lo: " << w.lopf() << endl;
		os << "hi: " << w.hipf();
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
					-0.12940952255092145,
					0.22414386804185735,
					0.83651630373746899,
					0.48296291314469025
				});
    }

	static const Wavelet<T> db7() {
        return Wavelet<T>("db7",
        		evector<T> {
					0.00035371380000103988,
					-0.0018016407039998328,
					0.00042957797300470274,
					0.012550998556013784,
					-0.01657454163101562,
					-0.038029936935034633,
					0.080612609151065898,
					0.071309219267050042,
					-0.22403618499416572,
					-0.14390600392910627,
					0.4697822874053586,
					0.72913209084655506,
					0.39653931948230575,
					0.077852054085062364
					});
    }

	static const Wavelet<T> sym3() {
		return Wavelet<T>("sym3",
				evector<T>{
				  0.035226291882100656,
				  -0.08544127388224149,
				  -0.13501102001039084,
				  0.4598775021193313,
				  0.8068915093133388,
				  0.3326705529509569
				  });
	}

	static const Wavelet<T> coif4() {
		return Wavelet<T>("coif4",
				evector<T>{
					-1.7849909144933469e-06,
					-3.259647940030751e-06,
					3.1229861599195265e-05,
					6.233885431278719e-05,
					-0.0002599743371222568,
					-0.0005890202246332165,
					0.0012665610789256603,
					0.0037514346971460866,
					-0.0056582838001308835,
					-0.015211728187697211,
					0.02508225333794961,
					0.03933442260558915,
					-0.09622042453595264,
					-0.06662747236681717,
					0.43438603311435653,
					0.7822389344242826,
					0.41530842700068227,
					-0.05607731960356926,
					-0.08126671024919373,
					0.02668230466960483,
					0.01606894713157503,
					-0.007346167936268051,
					-0.001629492425226786,
					0.000892313902537003
				  });
	}

	static const Wavelet<T> bior6_8() {
		return Wavelet<T>("bior6.8",
				evector<T>{
				  0.0,
				  0.0019088317364812906,
				  -0.0019142861290887667,
				  -0.016990639867602342,
				  0.01193456527972926,
				  0.04973290349094079,
				  -0.07726317316720414,
				  -0.09405920349573646,
				  0.4207962846098268,
				  0.8259229974584023,
				  0.4207962846098268,
				  -0.09405920349573646,
				  -0.07726317316720414,
				  0.04973290349094079,
				  0.01193456527972926,
				  -0.016990639867602342,
				  -0.0019142861290887667,
				  0.0019088317364812906
				  },
			    evector<T>{
				  0.0,
				  0.0,
				  0.0,
				  0.014426282505624435,
				  -0.014467504896790148,
				  -0.07872200106262882,
				  0.04036797903033992,
				  0.41784910915027457,
				  -0.7589077294536541,
				  0.41784910915027457,
				  0.04036797903033992,
				  -0.07872200106262882,
				  -0.014467504896790148,
				  0.014426282505624435,
				  0.0,
				  0.0,
				  0.0,
				  0.0
				  });
	}

	static const Wavelet<T> rbio6_8() {
		return Wavelet<T>("rbio6.8",
				evector<T>{
				  0.0,
				  0.0,
				  0.0,
				  0.0,
				  0.014426282505624435,
				  0.014467504896790148,
				  -0.07872200106262882,
				  -0.04036797903033992,
				  0.41784910915027457,
				  0.7589077294536541,
				  0.41784910915027457,
				  -0.04036797903033992,
				  -0.07872200106262882,
				  0.014467504896790148,
				  0.014426282505624435,
				  0.0,
				  0.0,
				  0.0
				  },
   			    evector<T>{
				  -0.0019088317364812906,
				  -0.0019142861290887667,
				  0.016990639867602342,
				  0.01193456527972926,
				  -0.04973290349094079,
				  -0.07726317316720414,
				  0.09405920349573646,
				  0.4207962846098268,
				  -0.8259229974584023,
				  0.4207962846098268,
				  0.09405920349573646,
				  -0.07726317316720414,
				  -0.04973290349094079,
				  0.01193456527972926,
				  0.016990639867602342,
				  -0.0019142861290887667,
				  -0.0019088317364812906,
				  0.0
				  });
	}
};

#endif //INC_04_WAVELETS_WAVELET_HPP
