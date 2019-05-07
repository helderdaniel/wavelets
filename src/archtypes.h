//
// Created by hdaniel on 02/05/19.
//

/**
 * define base architecture types to choose implementation
 * on algorithm classes
 */

#ifndef __ARCHTYPES_H__
#define __ARCHTYPES_H__

class SEQ {};	//Single core
class PAR {};	//Multi-core (multi-CPU) (OPENMP)
class PARa {};	//Alternate Multi-core (multi-CPU) (manual THREADS)
//class MPI {};	//Message-passing clusters with MPI
class GPU {};	//GPU

#endif //__ARCHTYPES_H__