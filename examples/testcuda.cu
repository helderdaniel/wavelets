//
// Created by hdaniel on 14/05/19.
//

/**
 * Simple test and example of using cudadevices.hpp, Cuda class
 *
 */


//#define NDEBUG  //At top before cudadevice.hpp or others that use it
#include "../src/cudadevices.hpp"

#include <iostream>

using namespace std;


int main() {
Cuda c;

	cout << c << endl;
	return 0;
}
