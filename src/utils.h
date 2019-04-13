//
// Created by hdaniel on 13/04/19.
//

#ifndef INC_04_WAVELIB_BENCH_UTILS_H
#define INC_04_WAVELIB_BENCH_UTILS_H

#include <ostream>
#include <vector>

using namespace std;

template <typename T>
ostream& operator<<(ostream& os, const vector<T>& v) {
    os << "[ ";
    for (int i=0; i < v.size(); ++i)
        os << v[i] << ' ';
    os << ']';
    return os;
}

template <typename T>
string toString(const vector<T>& v) {
    stringstream out;
    out << v;
    return out.str();
}

/**
 * Extend signal to handle wavelet around boundaries
 * Symmetric extension
 * based on: http://wavelet2d.sourceforge.net/ symm_ext()
 */
void symmExt(vector<double> &sig, int l) {
    unsigned int len = sig.size();

    for (int i =0; i < l; i++) {
        double temp1= sig[i * 2];
        double temp2= sig[len - 1];
        sig.insert(sig.begin(),temp1);
        sig.insert(sig.end(),temp2);
    }
}

#endif //INC_04_WAVELIB_BENCH_UTILS_H
