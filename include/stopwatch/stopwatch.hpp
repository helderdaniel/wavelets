/**
 * StopWatch class definition
 * v2.0 hdaniel@ualg.pt 2011
 * v2.1 hdaniel@ualg.pt 2019 apr
 * v3.0 hdaniel@ualg.pt 2019 apr C++14
 * Changelog:
 *      return time() from last reset() to last lap()
 *      Note:  Time is always real time, so it counts user inputs and event waits
 */

#include <chrono>
#include <iostream>
using namespace std;
using namespace::std::chrono;


class StopWatch {
time_point<high_resolution_clock> start, stop;
double elapsedTime;  //real elapsed time in seconds

public:
	/**
	 * Creates StopWatch object and resets timer
	 */
	StopWatch() { reset(); }

	/**
	 * sets start and stop times equals to current time
	 */
	void reset() {
		elapsedTime = 0;
		start = stop = high_resolution_clock::now();
	}

	/**
	 * sets stop times equals to current time
	 */
	void lap() {
		stop = high_resolution_clock::now();
		elapsedTime = duration<double>(stop - start).count();
	}

	/**
	 * Returns time elapsed since last reset() until last lap()
	 * DOES count user input like wait keypress
	 */
	double watch() { return elapsedTime; }

	//friend ostream& operator << (ostream& os, StopWatch& c);
};

/**
 * Prints real elapsed time in seconds
 */
ostream& operator << (ostream &os, StopWatch& sw) {
	os << sw.watch() << "s";
	return os;	
}
