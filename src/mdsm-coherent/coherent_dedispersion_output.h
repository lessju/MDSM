#ifndef COHERENT_OUTPUT_H_
#define COHERENT_OUTPUT_H_

#include "pthread.h"
#include "observation.h"
#include "unistd.h"
#include "cutil_inline.h"

typedef struct {
    // Input parameters
    int nthreads, iterations, maxiters;

    // Input and output buffers memory pointers
    cufftComplex* host_odata;
    float *       host_profile;
    size_t        host_osize;
   
    // Thread-specific info + synchronisation objects
    unsigned short stop;
    pthread_rwlock_t  *rw_lock;
    pthread_barrier_t *input_barrier;
    pthread_barrier_t *output_barrier;

    // Timing
    time_t start;

    // Survey parameters
    OBSERVATION *obs;

} OUTPUT_PARAMS;

void* process_output(void* output_params);

#endif
