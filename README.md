# Simple 1D wavelets lib for C++
Some fast wavelet processing implementation.

Currently implemented algorithms:
- Discrete wavelet transform

Implemented wavelets so far:
- Haar1, db1 (Haar1), db2, db7
- sym3
- coif4
- bior6.8, rbio6.8

### Example of use
Example below (available at examples folder) calls Discrete Wavelet Transform on input signal.
Compile it with at least g++ v7 and make sure c++17 standard is specified:

> g++-7 waveletdemo.cpp -o waveletdemo -std=c++17

```c++
//This is waveletdemo.cpp from examples/ folder

//include wavelet and extended vector
#include "../src/evector.hpp"
#include "../src/wavelet.hpp"
#include "../src/wavelettransform.hpp"

#include <iostream>
using namespace std;

#define DTYPE double

int main() {
	//define Wavelet here: haar1, db1, db2, db7
	Wavelet<DTYPE> wvlt = WaveletFactory<DTYPE>::db7();

	//input signal
	evector<DTYPE> input = { 32, 10, 20, 38, 37, 28, 38, 34, 18, 24, 18, 9, 23, 24, 28, 34 };

	//Dim output signal vector according to Wavelet coefficients
	const int wvltcoefs = wvlt.size();
	const int osize = WaveletTransform::outputSize(input.size(), wvltcoefs);
	auto output = evector<DTYPE>(2 * osize);

	//Perform transform
	WaveletTransform::dwt<DTYPE>(wvlt, input, output);

	cout << "Input signal:\n" << input << endl << endl;
	cout << "Discrete Wavelet Transform (" + wvlt.name() + "):\n" << output << endl;

	return 0;
}
```

It should display:

```
Input signal:
[ 32 10 20 38 37 28 38 34 18 24 18 9 23 24 28 34 ]

Discrete Wavelet Transform (db7):
[ 23.6753 43.1243 47.6845 47.4638 29.454 33.2002 31.8586 46.3168 50.7474 38.5833 22.8656 24.84 39.7161 46.2875 -12.0386 17.2375 -13.2662 13.1351 -7.52492 -4.17754 9.21827 -0.83275 -0.956452 1.87084 -6.60709 -1.54342 9.6809 -6.64522 ]
```

## Related works
The C++ implementation in this repo, follows the Wavelet coefficients as defined in the Python module: **pywt**.

https://www.pybytes.com/pywavelets/

Note that the coefficients reported in:

http://wavelets.pybytes.com/

are NOT correct with the **pywt** module.

To check the coefficients of the Python module was used the snippet:

```
wavelets = ['haar', 'db7', 'sym3', 'coif4', 'bior6.8', 'rbio6.8' ]  #...
w = wavelets[1] #select db7 wavelet

print(pywt.Wavelet(w).dec_lo)   #get low  pass filter coefficients
print(pywt.Wavelet(w).dec_hi)   #get high pass filter coefficients
```

The C++ implementation defined in:

http://wavelet2d.sourceforge.net/

have different coefficients for some wavelets.
This lib uses fftw3 lib:

http://www.fftw.org/

for Fast Fourier Transforms.


