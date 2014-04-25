#ifndef THREAD_H_
#define THREAD_H_

#include "pthread.h"
#include "survey.h" 
#include "unistd.h"
#include "cufft.h"
#include <omp.h>

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

typedef struct 
{
	// Overall specs
	int num_devices;
	unsigned long int minTotalGlobalMem; // When assuming homogeneous devics

	DEVICE_INFO* devices;

} DEVICES;

typedef struct THREAD_PARAMS
{
    // Input parameters
    unsigned iterations; // The number of iteration a thread has to wait to begin processing (main body)
    unsigned maxiters;   // The number of iteration to wait during for shutdown

    // Input and output buffers memory pointers & sizes
    size_t inputsize, outputsize;
    float **output;
    unsigned char *input; 

    // GPU information
    unsigned device_id;

    // Thread-specific info + synchronisation objects
    unsigned short thread_num, num_threads, stop;
    pthread_rwlock_t  *rw_lock;
    pthread_barrier_t *input_barrier;
    pthread_barrier_t *output_barrier;

    // Timing
    time_t start;

    // Survey parameters
    SURVEY* survey;

} THREAD_PARAMS;

void allocateBuffer(void **pointer, size_t size);

#endif
