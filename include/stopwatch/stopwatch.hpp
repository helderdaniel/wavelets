/**
 * StopWatch class definition
 * v2.0 hdaniel@ualg.pt 2011
 * v2.1 hdaniel@ualg.pt 2019 apr
 * Changelog:
 *      return cpuTime() and realTime() from last reset() to last lap()
 *      Note0: cpuTime() is read first, so should be a little slower, even if realTime()
 *             measure was not delayed by waiting for some event.
 *		Note1: unlike previous versions cpuTime() and realTime() do not call lap()
 *		Note2: Current implementation of realTime() has miliseconds resolution.
 *		       std::chrono can be used in C++14 to get real time with higher resolution
 *		       (however duration.count() is an integer):
 *
 *                 #include <chrono>
 *                 using namespace::std::chrono
 *                 //...
 *		           auto start = high_resolution_clock::now();
 *		           //...
 *		           auto stop = high_resolution_clock::now();
 *		           auto duration = duration_cast<nanoseconds>(stop - start);
 *		           cout << duration.count() << endl;
 *
*/

#include <sys/timeb.h>
#include <time.h>
#include <iostream>
using namespace std;

class StopWatch {
struct timeb starttm, endtm;
clock_t start, end;

public:
	/**
	 * Creates StopWatch object and resests timers
	 */
	StopWatch() { reset(); }

	/**
	 * sets start and end times equals to current time
	 */
	void reset()	{
		ftime(&starttm);
		endtm = starttm;
		start = end = clock();
	}

	/**
	 * sets end times equals to current time
	 */
	void lap() {
        end=clock();
		ftime(&endtm);
	}

	/**
	 * Returns time elapsed since last reset() until last lap()
	 * DOES count user input like wait keypress
	 */
	double realTime() {
		const time_t startsec = starttm.time;
		const time_t endsec = endtm.time;
		const short startms = starttm.millitm;	
		const short endms = endtm.millitm;
		return (endsec-startsec)+((double)(endms-startms)/1000);
	};

	/**
	 * Returns time elapsed since last reset() until last lap()
	 * DOES NOT count user input like wait keypress
	 * (just processor ticks)
     */
	double cpuTime() {
		return ((double)(end-start)/CLOCKS_PER_SEC);
	};
	//friend ostream& operator << (ostream& os, StopWatch& c);
};

/**
 * Prints: "realTime (cpuTime)" in seconds"
 * in *.cpp in previous versions
 */
ostream& operator << (ostream &os, StopWatch& sw) {
	os << sw.realTime() << "s (" << sw.cpuTime() << "s) ";
	return os;	
}
