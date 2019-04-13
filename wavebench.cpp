/**
 * Simple dwt db7 benchmark for wavelib
 * 
 * hdaniel@ualg.pt apr 2019
 */
/*
read matlab
https://github.com/hbristow/cvmatio (does not write)
https://sourceforge.net/projects/matio/


create class:
wavelet class hierarchy
waveletTransform class:
    dwt()
    packetTransform()

class must be generic to accept double, float
*/

#include <iostream>
#include <fstream>
#include <vector>
//#include <cmath>
#include "stopwatch/stopwatch.h"
#include "../00-wavelibsrc/wavelet2s.h"

using namespace std;

template <typename T>
ostream& operator<<(ostream& os, const vector<T>& v) {   
	os << "[ ";
	for (int i=0; i < v.size(); ++i)
    		os << v[i] << ' ';
	os << ']';
	return os;
}

/**
 * Extend signal to handle wavelet around boundaries
 * Symetric extension
 * from: http://wavelet2d.sourceforge.net/
 */
/*
void* symm_ext(vector<double> &sig, int l) {
	unsigned int len = sig.size();

	for (int i =0; i < l; i++) {
		double temp1= sig[i * 2];
		double temp2= sig[len - 1];
		sig.insert(sig.begin(),temp1);
		sig.insert(sig.end(),temp2);
	}

	return 0;
}
*/

struct db7wl {
    //Approximation, Low pass filter db7 coeficients
    float a[14] = {
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
    };

    //Detail, High pass filter db7 coeficients
    float d[14] = {
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
    };
} db7;

//dwt
void dwtp(const float (&w)[14], float* input, vector<float> &output) {
const int coefsize = size(w);
    
    for (int i=0; i<coefsize*2; i+=2) {
        float t=0;
        for (int j=0; j<coefsize; ++j) {
            t += input[i+j] * w[j]; 
        }
        output.push_back(t);
    }
}

void dwt(const db7wl &w, vector<double> &input, vector<float> &output) {
const int coefsize = size(w.a);

    //extend signal to avoid boundaries distortion
    symm_ext(input, coefsize-2);
    
    //dwt low pass
    dwtp(w.a, (float *) input.data(), output);
    dwtp(w.d, (float *) input.data(), output);
}

int main() {
StopWatch sw;
vector<float> output1;
vector<double> output2, flag;
vector<int> length;

const int times = 10000;

    sw.reset();
    for (int i=0; i<times; ++i) {
        vector<double> signal = { 32, 10, 20, 38, 37, 28, 38, 34, 18, 24, 18, 9, 23, 24, 28, 34 };
        dwt(db7, signal, output1);
    }
    sw.lap();
    cout << sw << endl;

    sw.reset();
    for (int i=0; i<times; ++i) {
        vector<double> signal = { 32, 10, 20, 38, 37, 28, 38, 34, 18, 24, 18, 9, 23, 24, 28, 34 };
	    dwt_sym(signal, 1, "db7", output2, flag, length);
    }        
    sw.lap();
    cout << sw << endl;

    cout << output1.size() << endl;
    cout << output2.size() << endl;

    //dif by sum of squared errors (SSE)
    double diff=0;
    for (int i=0; i<output1.size(); ++i)
        diff += pow(output1[i]-output2[i], 2);
    cout << diff/output1.size() << endl;

    ofstream out1("out1.txt");
	ofstream out2("out2.txt");
    out1 << output1 << endl;
    out2 << output2 << endl;

    return 0;
}