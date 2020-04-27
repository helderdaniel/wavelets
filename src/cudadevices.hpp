//
// Created by hdaniel on 14/05/19.
//

#ifndef __CUDADEVICE_HPP__
#define __CUDADEVICE_HPP__

#include "cudacheck.h"
#include <vector>
#include <map>
//#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <iomanip>
using namespace std;

/**
 *  Describes Cuda devices on host
 *  allows communication with the devices
 *  and launching threads on them
 */


class Cuda {
const static map<int, int> _coresMP;
int _availdevices;
int _currentdevice;

static int coresMP(int major, int minor);

public:
	Cuda() {
		checkCUDA(cudaGetDevice(& _currentdevice));
		checkCUDA(cudaGetDeviceCount (& _availdevices));
	}

	/**
	 * @return: current device selected
	 */
	int curDevice() const { return _currentdevice; }

	/**
	 * @return: number of available devices
	 */
	int numDevices() const { return _availdevices; }

	/**
	 * @param dev: device to select
	 */
	void selDevice(int dev) {
		checkCUDA(cudaSetDevice(dev));
		_currentdevice = dev;
	}

	cudaDeviceProp info(const int device) const {
		cudaDeviceProp prop;
		checkCUDA(cudaGetDeviceProperties(&prop, device));
		return prop;
	}

	friend ostream& operator<<(ostream& os, const Cuda& c);
};


/* Cuda cores by compute capabilities as in:
 *
 * cuda/samples/common/inc/helper_cuda.h/_ConvertSMVer2Cores(major,minor)
 * Note: OLD SM 2.x arch is deprecated
*/
const map<int, int>Cuda::_coresMP = {

		{ 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
		{ 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
		{ 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
		{ 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
		{ 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
		{ 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
		{ 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
		{ 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
		{ 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
		{ 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
		{ 0x70, 64 }, // Volta Generation (SM 7.0) GV100 class
};


int Cuda::coresMP(int major, int minor) {
	const int version = (major<<4)+minor; //in hexa as map below
	if (_coresMP.find(version) == _coresMP.end()) return -1;
	return _coresMP.at(version);  //_coresMP[version] did not worked! why?
}

ostream& operator<<(ostream& os, const Cuda& c) {
	int driverVersion = 0, runtimeVersion = 0;
	vector<cudaDeviceProp> devices(c.numDevices());

	checkCUDA(cudaDriverGetVersion(&driverVersion));
	checkCUDA(cudaRuntimeGetVersion(&runtimeVersion));
	for (int i=0; i<devices.size(); ++i)
		devices[i] = c.info(i);

	os << "Cuda driver version:  " << driverVersion/1000 << "." << (driverVersion%100)/10 << endl;
	os << "Cuda runtime version: " << runtimeVersion/1000 << "." << (runtimeVersion%100)/10 << endl;
	os << endl;
	os << "Cuda devices present: " << devices.size() << endl;
	os << "Selected device:      " << c.curDevice() << endl;
	os << endl;
	for (int i=0; i<devices.size(); ++i) {
		os << "Device " << i << endl;
		os << "GPU Device name:      " << devices[i].name << endl;
		os << "Compute capability:   " << devices[i].major << "." << devices[i].minor << endl;

		//0 is the default: multiple threads can use the device
		//2 means the no threads can use the device
		//https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7eb25f5413a962faad0956d92bae10d0
		os << "Compute mode:         " << devices[i].computeMode << endl;

		os << "MultiProcessors:      " << devices[i].multiProcessorCount << endl;
		os << "Cuda cores by SMP:    " << Cuda::coresMP(devices[i].major, devices[i].minor)  << endl;
		os << "Total Cuda cores:     " << Cuda::coresMP(devices[i].major, devices[i].minor) * devices[i].multiProcessorCount << endl;

		os << "Warp size:            " << devices[i].warpSize << endl;
		os << "Registers per block:  " << devices[i].regsPerBlock << endl;

		os << "Global memory:        " << setfill(' ')  << setw(10) << devices[i].totalGlobalMem << " Bytes" << endl;
		os << "Constant memory:      " << setw(10) << devices[i].totalConstMem  << " Bytes" << endl;
		os << "Shared mem per block: " << setw(10) << devices[i].sharedMemPerBlock  << " Bytes" << endl;

		os << "ECC enabled:          " << (devices[i]. ECCEnabled ? "yes" : "no") << endl;
		os << "Can map host memory:  " << (devices[i].canMapHostMemory ? "yes" : "no") << endl;

		os << "Unified Address(UVA): " << (devices[i].unifiedAddressing ? "yes" : "no") << endl;
		os << "Concurrent copy and kernel execution: " << (devices[i].deviceOverlap ? "yes" : "no");
		os << " with " << devices[i].asyncEngineCount << " copy engine(s)" << endl;
	}
	return os;
}


#endif //__CUDADEVICE_HPP__
