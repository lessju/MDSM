#ifndef OUTPUT_H_
#define OUTPUT_H_

#include "multibeam_dedispersion_writer.h"
#include "pthread.h"
#include "survey.h"
#include "unistd.h"

#define MDSM_STAGES 3

typedef struct {
    // Input parameters
    unsigned nthreads, iterations, maxiters;

    // Separate output buffer per beam
    float  **output_buffer, **input_buffer;
   
    // Thread-specific info + synchronisation objects
    unsigned short stop;
    pthread_rwlock_t  *rw_lock;
    pthread_barrier_t *input_barrier;
    pthread_barrier_t *output_barrier;

    // Writer-related objects
    WRITER_PARAMS     *writer_params;
    pthread_mutex_t   *writer_mutex;
    float             *writer_buffer;

    // Timing
    time_t start;

    // Survey parameters
    SURVEY *survey;

} OUTPUT_PARAMS;

void* process_output(void* output_params);

#endif
