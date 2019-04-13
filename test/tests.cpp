//
// Created by hdaniel on 13/04/19.
//
#include "catch2/catch.hpp"
#include <vector>
#include <sstream>

#include "../src/utils.h"

using namespace std;


TEST_CASE( "Symmetric extension", "[wavelet]" ) {
    string vout = "[ 1.1 2.2 3.3 4.4 5.5 ]";
    vector<double> v = {1.1, 2.2, 3.3, 4.4, 5.5};

    SECTION("Stream vector") {
        stringstream out;
        out << v;
        REQUIRE(out.str() == vout);
    }

    SECTION("Symmetric extension") {

    }

}

