#ifndef COHERENT_THREAD_H_
#define COHERENT_THREAD_H_

#include "cutil_inline.h"
#include "pthread.h"
#include "observation.h"
#include "unistd.h"

typedef struct {

    // Device capabilities
    int multiprocessor_count;
    int constant_memory;
    int shared_memory;
    int register_count;
    int thread_count;
    int clock_rate;
    int device_id;

} DEVICE_INFO;

typedef struct {
	// Overall specs
	int num_devices;
	unsigned long int minTotalGlobalMem; // When assuming homogeneous devics

	DEVICE_INFO* devices;

} DEVICES;

typedef struct {
    // Input parameters
    int iterations, maxiters;

    // Input and output buffers memory pointers & sizes
    size_t host_isize, host_osize, device_isize, device_osize, profile_size;
    cufftComplex* host_odata;
    cufftComplex* host_idata;
    float*        host_profile; 
   
    // Thread-specific info + synchronisation objects
    int device_id;
    unsigned short thread_num, num_threads, stop;
    pthread_rwlock_t  *rw_lock;
    pthread_barrier_t *input_barrier;
    pthread_barrier_t *output_barrier;

    // Timing
    time_t start;

    // Survey parameters
    OBSERVATION* obs;

} THREAD_PARAMS;

#endif
