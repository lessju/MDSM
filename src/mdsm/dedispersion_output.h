#ifndef OUTPUT_H_
#define OUTPUT_H_

#include "pthread.h"
#include "survey.h"
#include "unistd.h"

typedef struct {
    // Input parameters
    int nthreads, iterations, maxiters;

    // Input and output buffers memory pointers
    float* output_buffer;
    size_t dedispersed_size;
   
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
