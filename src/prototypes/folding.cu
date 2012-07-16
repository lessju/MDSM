#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "unistd.h"
#include "time.h"
#include "string.h"
#include <cutil_inline.h>

// ---------------------- Optimised Dedispersion Loop  ------------------------------
__global__ void fold(float *input, float *output, int nsamp, float tsamp,
                    float period)
{
    int bins = period / tsamp;

    for(unsigned b = threadIdx.x;
                 b < bins;
                 b += blockDim.x)
    {
        float val = 0;
        for(unsigned s = 0; s < nsamp / bins; s ++)
            val += input[blockIdx.x * nsamp + s * bins + b];
         output[blockIdx.x * bins + b] = val;
    }

}

// -------------------------- Main Program -----------------------------------

float period = 64, tsamp = 4;
int nsamp = 1024, tdms = 2;
int gridsize = 128, blocksize = 128;

// Process command-line parameters
void process_arguments(int argc, char *argv[])
{
    int i = 1;
    
    while((fopen(argv[i], "r")) != NULL)
        i++;

    while(i < argc) {
       if (!strcmp(argv[i], "-nsamp"))
           nsamp = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-tdms"))
           tdms = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-gridsize"))
           gridsize = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-blocksize"))
           blocksize = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-tsamp"))
           tsamp = atof(argv[++i]);
       else if (!strcmp(argv[i], "-period"))
           period = atof(argv[++i]);
       i++;
    }
}

int main(int argc, char *argv[])
{
   float *input, *output, *d_input, *d_output;
   int i, j;

   process_arguments(argc, argv);

    // Calculate folding bins
    int bins = period / tsamp;

    // Allocate and initialise arrays
    input =  (float *) malloc( tdms * nsamp * sizeof(float));
    output = (float *) malloc( tdms * bins * sizeof(float));
    for(i = 0; i < tdms; i++)
        for(j = 0; j < nsamp; j++) {
            input[i *nsamp + j] = i + 1;
            printf("Dm: %d samp: %d = %f \n", i, j, input[i*nsamp+j]);
         }
         

    // Initialise CUDA stuff
    cutilSafeCall( cudaSetDevice(1));
    cudaEvent_t event_start, event_stop;
    float timestamp;

    cudaEventCreate(&event_start); 
    cudaEventCreate(&event_stop);

   printf("nsamp: %d, tdms: %d, tsamp: %f, period: %f, bins: %d\n",
           nsamp, tdms, tsamp, period, bins);

    // Allocate CUDA memory and copy dmshifts
    cutilSafeCall( cudaMalloc((void **) &d_input, tdms * nsamp * sizeof(float)));
    cutilSafeCall( cudaMalloc((void **) &d_output, bins * tdms * sizeof(float)));

    time_t start = time(NULL);

    // Copy input to GPU
    cudaEventRecord(event_start, 0);
    cutilSafeCall( cudaMemcpy(d_input, input, tdms * nsamp * sizeof(float), cudaMemcpyHostToDevice) );    
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied to GPU in: %lf\n", timestamp);

    cudaEventRecord(event_start, 0);
    fold<<<dim3(tdms, 1), blocksize>>>(d_input, d_output, nsamp, tsamp, period);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Folded in: %lf\n", timestamp);

    // Copy output from GPU
    cudaEventRecord(event_start, 0);
    cutilSafeCall( cudaMemcpy(output, d_output, bins * tdms * sizeof(float), cudaMemcpyDeviceToHost) );    
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied from GPU in: %lf\n", timestamp);

    // Check values
    for(i = 0; i < tdms; i++)
        for(j = 0; j < bins; j++)
            if (output[i*bins+j] != (i+1) * (nsamp / bins)) {
                printf("Dm: %d bin: %d = %f [%f] \n", i, j, output[i*bins+j], (i+1) * nsamp / (bins * 1.0));
                exit(-1);
            }

    printf("Total time: %d\n", (int) (time(NULL) - start));
}

