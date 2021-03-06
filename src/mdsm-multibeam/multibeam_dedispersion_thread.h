#ifndef THREAD_H_
#define THREAD_H_

#include "pthread.h"
#include "survey.h" 
#include "unistd.h"
#include <omp.h>

#define MDSM_STAGES 3

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

typedef struct
{
    unsigned device_id;
    unsigned num_threads;
    unsigned *thread_ids;
    unsigned primary_thread;
    pthread_barrier_t barrier;
} GPU;

typedef struct THREAD_PARAMS
{
    // Input parameters
    unsigned iterations; // The number of iteration a thread has to wait to begin processing (main body)
    unsigned maxiters;   // The number of iteration to wait during for shutdown

    // Input and output buffers memory pointers & sizes
    size_t inputsize, outputsize, antenna_size;
    unsigned char *antenna_buffer;
    float **output;
    float **input; 

    // GPU-specific
    float *d_input, *d_output;
    unsigned gpu_index; 
    GPU **gpus;

    // Thread-specific info + synchronisation objects
    unsigned short thread_num, num_threads, stop;
    pthread_rwlock_t  *rw_lock;
    pthread_barrier_t *input_barrier;
    pthread_barrier_t *output_barrier;
    pthread_barrier_t *gpu_barrier;

    // Timing
    time_t start;

    // Survey parameters
    SURVEY* survey;

    // Pointer to all threads in the system
    THREAD_PARAMS **cpu_threads;

} THREAD_PARAMS;

void allocateInputBuffer(float **pointer, size_t size);
void allocateOutputBuffer(float **pointer, size_t size);


#endif
