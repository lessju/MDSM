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
    int iterations, maxiters;

    // Dedispersion parameters
    int maxshift, binsize;
    float *dmshifts;

    // Input and output buffers memory pointers & sizes
    size_t inputsize, outputsize, dm_output;
    float* output;
    float* input; 
   
    // Thread-specific info + synchronisation objects
    int device_id;
    unsigned short thread_num, num_threads, stop;
    pthread_rwlock_t  *rw_lock;
    pthread_barrier_t *input_barrier;
    pthread_barrier_t *output_barrier;

    // Timing
    time_t start;

    // Survey parameters
    SURVEY* survey;

} THREAD_PARAMS;

#endif
