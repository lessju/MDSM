#ifndef OUTPUT_H_
#define OUTPUT_H_

#include "pthread.h"
#include "survey.h"
#include "unistd.h"

typedef struct {

    // Input parameters
    unsigned nthreads, iterations, maxiters;

    // Separate output buffer per beam
    float  **output_buffer;
   
    // Thread-specific info + synchronisation objects
    unsigned short stop;
    pthread_rwlock_t  *rw_lock;
    pthread_barrier_t *input_barrier;
    pthread_barrier_t *output_barrier;

    // Timing
    time_t start;

    // Survey parameters
    SURVEY *survey;

} OUTPUT_PARAMS;

void* process_output(void* output_params);

#endif
