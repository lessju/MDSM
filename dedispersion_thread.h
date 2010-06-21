#ifndef THREAD_H_
#define THREAD_H_

#include "cutil_inline.h"
#include "pthread.h"
#include "survey.h"
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
    // Input parameters
    int nchans, nsamp, iterations, maxiters;
    float tsamp;

    // Dedispersion parameters
    float *dmshifts;
    float *dmvalues, startdm, dmstep;
    int ndms, maxshift, binsize;

    // Input and output buffers memory pointers & sizes
    size_t inputsize, outputsize;
    float* output;
    float* input; 
   
    // Thread-specific info + synchronisation objects
    int device_id;
    unsigned short thread_num, stop;
    pthread_rwlock_t  *rw_lock;
    pthread_barrier_t *input_barrier;
    pthread_barrier_t *output_barrier;

    // Timing
    time_t start;

    // Survey parameters
    SURVEY* survey;

} THREAD_PARAMS;

// void* dedisperse(void* thread_params);
// DEVICE_INFO** initialise_devices(int *num_devices);

#endif
