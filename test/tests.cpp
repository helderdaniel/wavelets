//
// Created by hdaniel on 13/04/19.
//
#include "catch2/catch.hpp"
#include <vector>
#include <sstream>

#include "../src/utils.h"

using namespace std;

TEST_CASE( "Vector operations", "[vector]" ) {
    string v0str = "[ 1.1 2.2 3.3 4.4 5.5 ]";
    vector<double> v0 = {1.1, 2.2, 3.3, 4.4, 5.5};

    SECTION("Stream vector") {
        stringstream out;
        out << v0;
        REQUIRE(out.str() == v0str);
    }

    SECTION("Vector to string") {
        stringstream out;
        out << v0;
        REQUIRE(toString(v0) == v0str);
    }

    SECTION("Symmetric extension") {
        vector<double> v1 = v0; //copies vector

        symmExt(v1, 3);
        REQUIRE(toString(v1) == "[ 3.3 2.2 1.1 1.1 2.2 3.3 4.4 5.5 5.5 4.4 3.3 ]");
    }

}

