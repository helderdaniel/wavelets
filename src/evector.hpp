//
// Created by hdaniel on 20/04/19.
//

#ifndef INC_04_WAVELIB_BENCH_EVECTOR_HPP
#define INC_04_WAVELIB_BENCH_EVECTOR_HPP


/*
 * JLBorges
 * http://www.cplusplus.com/forum/general/156923/
 *
 * taken from:
 *
 * Stroustrup in A Tour of C++ (draft)
 * http://isocpp.org/files/papers/4-Tour-Algo-draft.pdf
 * pag. 104
 *
 * It is ok to inherit publicly from std::vector<> as long as:

a. We remember that every operation provided by std::vector<> must be a semantically valid
   operation on an object of the derived class

b. We avoid creating derived class objects with dynamic storage duration.
   (std::vector<> does not have a virtual destructor).

The standard library vector does not guarantee range checking. For example:

vector<Entry> phone_book(1000);
int i = phone_book[2001].number; // 2001 is out of range

That initialization is likely to place some random value in i rather than giving an error.
This is undesirable and out-of-range errors are a common problem.
Consequently, I often use a simple range-checking adaptation of vector:

template<typename T>
class Vec : public std::vector<T> {
public:
    using vector<T>::vector; // use the constructors from vector

    T& operator[](int i) { return vector<T>::at(i); } // range-checked

    const T& operator[](int i) const { return vector<T>::at(i); } // range-checked
    // for const objects;
};

Vec inherits everything from vector except for the subscript operations that it redefines to do range checking. The at() operation is a vector subscript operation that throws an exception of type out_of_range if its argument is out of the vector’s range.

An out-of-range access will throw an exception that the user can catch.
*/


#include <ostream>
#include <sstream>
#include <iomanip>
#include <vector>

using namespace std;

template<typename T>
class evector : public vector<T> {
	static constexpr char defaultSeparator = ' ';
	static constexpr int defaultPrecision = -1;
	static constexpr int defaultFixedPrecision = -1;

public:
	//NOTE: does NOT convert from vector to evector
	using vector<T>::vector; //use the constructors from vector

	/**
	 * Extend signal to handle wavelet around boundaries
	 * Symmetric extension
  	 */
	void symmExt(int l) {
		unsigned int len = this->size();

		this->insert(this->begin(), this->rend()-l, this->rend());
		this->insert(this->end(), this->rbegin(), this->rbegin()+l);
		/*
		implementation of: http://wavelet2d.sourceforge.net/ symm_ext()
		for (int i =0; i < l; i++) {
			//rewrite with out cycle
			T temp1 = (*this)[i * 2];
			T temp2 = (*this)[len - 1];
			this->insert(this->begin(), temp1);
			this->insert(this->end(), temp2);
		}
		*/
	}

	/**
	 * @return average of vector
	 */
	double avg() {
		return accumulate(this->begin(), this->end(), 0.0) / this->size();
	}

	string toString(char sep=' ', int prec=defaultPrecision, int fixedPrec=defaultFixedPrecision) const {
		stringstream os;
		//override default precisions
		if (prec>=0) os << setprecision(prec);
		if (fixedPrec>=0) os << fixed << setprecision(fixedPrec);

		os << "[" << sep;
		for (int i=0; i < this->size(); ++i)
			os << (*this)[i] << sep;
		os << "]";
		return os.str();
	}

	//Could use Named Parameter Idiom
	//https://isocpp.org/wiki/faq/ctors#named-parameter-idiom
	//see also project namedParIdiom on these folders
	string toString(int prec) const { return toString(defaultSeparator, prec); }
	//Cannot do the next overload, cause signature is equal to previous overloaded func: toString(int)
	//string toString(int fixedPrec) { return toString(defaultSeparator, defaultPrecision, fixedPrec); }

	//friend ostream& operator<<(ostream& os, const evector<T>& v);
};


template <typename T>
ostream& operator<<(ostream& os, const evector<T>& v) {
	os << v.toString();
	return os;
}

#endif //INC_04_WAVELIB_BENCH_EVECTOR_H