#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "unistd.h"
#include "time.h"
#include "string.h"
#include <cuda_runtime.h>

// ---------------------- Optimised Dedispersion Loop  ------------------------------
__global__ void testAtomicCas(int *buffer, int nsamp, int factor)
{

    if (blockIdx.x * blockDim.x + threadIdx.x > nsamp * factor)
        return;

    // Get memory index
    int ind = (blockIdx.x * blockDim.x + threadIdx.x) / factor; 

    // Perform atomic CAS
    while (atomicCAS(buffer + ind, 0 , 1) == 0);

}

// -------------------------- Main Program -----------------------------------

int nsamp = 1024, blocksize = 128, factor = 32;

// Process command-line parameters
void process_arguments(int argc, char *argv[])
{
    int i = 1;
    
    while((fopen(argv[i], "r")) != NULL)
        i++;

    while(i < argc) {
       if (!strcmp(argv[i], "-nsamp"))
           nsamp = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-blocksize"))
           blocksize = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-factor"))
           factor = atoi(argv[++i]);
       i++;
    }
}


int main(int argc, char *argv[])
{
    int *buffer, *d_buffer;
    int i;

    process_arguments(argc, argv);

    // Allocate and initialise arrays
    buffer = (int *) malloc(nsamp * sizeof(int));

    // Initialise CUDA stuff
    cudaSetDevice(0);
    cudaEvent_t event_start, event_stop;
    float timestamp;

    cudaEventCreate(&event_start); 
    cudaEventCreate(&event_stop);

    printf("nsamp: %d, blocksize: %d, factor: %d\n", nsamp, blocksize, factor);

    // Allocate CUDA memory
    cudaMalloc((void **) &d_buffer, nsamp * sizeof(int));
    cudaMemset(d_buffer, 0, nsamp * sizeof(int));

    time_t start = time(NULL);

    // Launch GPU kernel
    dim3 gridDim(factor * nsamp / blocksize, 1);  
    cudaEventRecord(event_start, 0);
    testAtomicCas<<<gridDim, blocksize>>>(d_buffer, nsamp, factor);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Processed in: %lf\n", timestamp);

    // Copy output from GPU
    cudaEventRecord(event_start, 0);
    cudaMemcpy(buffer, d_buffer, nsamp * sizeof(int), cudaMemcpyDeviceToHost);    
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied from GPU in: %lf\n", timestamp);

    // Check values
    for(i = 0; i < nsamp; i++)
        if (buffer[i] != 1)
            printf("buffer[%d] = %d\n", i, buffer[i]);

    printf("Total time: %d\n", (int) (time(NULL) - start));
}

