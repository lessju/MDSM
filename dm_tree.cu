#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cutil_inline.h>

int nchans = 2048, nsamp = 128 * 1024;

__shared__ float input_shared[2000];
__shared__ float output_shared[2000];

// Uses shared memory to load and store values
__global__ void optimised_tree(float *input, float *output, int nsamp, int nchans, int shift, int gsize)
{
    // Assume #threads is #samples that fit in shared memory
    int s, c, g, s2, c1, c2, samp, limit, shareLimit, i;

    limit = nsamp + nchans - gsize;
    s = threadIdx.x + blockIdx.x * blockDim.x;
    for(samp = 0; samp + s < limit; samp += blockDim.x * gridDim.x) {
         
        __syncthreads();
        
        // Read data from global memory into shared memory
        shareLimit = (limit - samp - blockDim.x * blockIdx.x) * nchans;
        for (i = 0; i + threadIdx.x < min(shareLimit, nchans * (blockDim.x + gsize) ); i += blockDim.x)
            input_shared[i + threadIdx.x] = input[ (samp + blockDim.x * blockIdx.x) * nchans + i + threadIdx.x];

        __syncthreads();

        // Loop over group channels
        for(c = 1; c < gsize + 1; c++) {

            s2 = s + (shift * c / 2);

            // Loop over all groups
            for(g = 0; g < nchans; g = g + gsize) {
                    
                c1 = g + (c - 0.5) / 2;
                c2 = c1 + gsize / 2;

                output_shared[s * nchans + g + c - 1] = input_shared[s * nchans + c1] + 
                                                              input_shared[s2 * nchans + c2];
            }
        }

        __syncthreads();

        // Write data from shared memory to global memory
        for (i = 0; i + threadIdx.x < nchans * blockDim.x; i += blockDim.x)
            output[ (samp + blockDim.x * blockIdx.x) * nchans + i + threadIdx.x] = output_shared[i + threadIdx.x];

        __syncthreads();
    }
}

// Unoptimised Tree De-dispersion Algorithm
__global__ void dedisperse_tree(float *input, float *output, int nsamp, int nchans, int shift, int gsize)
{
    int s, c, g, s2, c1, c2, samp, soffset, limit;

    // Loop samples
    limit = nsamp + nchans - gsize;
    soffset = threadIdx.x + blockIdx.x * blockDim.x;
    for(samp = 0; samp + soffset < limit; samp += blockDim.x * gridDim.x) {
           
        s = soffset + samp;

        // Loop over group channels
        for(c = 1; c < gsize + 1; c++) {

            s2 = s + (shift * c / 2);

            // Loop over all groups
            for(g = 0; g < nchans; g = g + gsize) {
                    
                c1 = g + (c - 0.5) / 2;
                c2 = c1 + gsize / 2;

                output[s * nchans + g + c - 1] = input[s * nchans + c1] + input[s2 * nchans + c2];
            }
        }
    }
}

// Process command-line parameters
void process_arguments(int argc, char *argv[])
{
    int i = 1;
    
    while((fopen(argv[i], "r")) != NULL)
        i++;

    while(i < argc) {
       if (!strcmp(argv[i], "-nchans"))
           nchans = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-nsamp"))
           nsamp = atoi(argv[++i]);
       i++;
    }
}

int main(int argc, char *argv[])
{
    float *input, *output, *d_input, *d_output;
    int nstages = (int) (log((float) nchans) / log(2.0) + 0.5), i, j, gsize, ns;

    process_arguments(argc, argv);
    printf("nsamp: %d, nchans: %d\n", nsamp, nchans);

    input = (float *) malloc( (nsamp + nchans) * nchans * sizeof(float));
    output = (float *) malloc( (nsamp + nchans) * nchans * sizeof(float));
    for (i = 0; i < nsamp + nchans; i++)
        for(j = 0; j < nchans; j++)
            input[i * nchans + j] = 1;

    cutilSafeCall( cudaSetDevice(0));
    cudaEvent_t event_start, event_stop;
    float timestamp;
    int gridsize = 1;
    dim3 gridDim(gridsize, 1);  

    cudaEventCreate(&event_start); 
    cudaEventCreate(&event_stop); 

    cutilSafeCall( cudaMalloc((void **) &d_input, (nsamp + nchans) * nchans * sizeof(float)));
    cutilSafeCall( cudaMalloc((void **) &d_output, (nsamp + nchans) * nchans * sizeof(float)));
    cutilSafeCall( cudaMemset(d_input, 0, (nsamp + nchans) * nchans * sizeof(float)));
    cutilSafeCall( cudaMemset(d_output, 0, (nsamp + nchans) * nchans * sizeof(float)));
    cutilSafeCall( cudaMemcpy(d_input, input, (nsamp + nchans) * nchans * sizeof(float), cudaMemcpyHostToDevice) );

    cudaEventRecord(event_start, 0);
    for(ns = 1; ns < nstages + 1;  ns++) {
        gsize = (int) pow(2.0, (float) ns);
        dedisperse_tree<<< gridDim, 64 >>>(d_input, d_output, nsamp, nchans, 1, gsize); 
        cutilSafeCall(cudaMemcpy(d_input, d_output, (nsamp + nchans) * nchans * sizeof(float), cudaMemcpyDeviceToDevice) );
    }
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Tree processed in: %lf\n", i, timestamp);

    cutilSafeCall(cudaMemcpy(output, d_output, (nsamp + nchans) * nchans * sizeof(float), cudaMemcpyDeviceToHost) );
    for(i = 0; i < nsamp; i++ )
        for (j = 0; j < nchans; j++)
            if ( output[i * nchans + j] != nchans);
                printf("Error: samp: %d, chan: %d, value: %d\n", i, j, (int) output[i * nchans + j]);

//    for(i = 0; i < nsamp; i++ ) {
//        for (j = 0; j < nchans; j++)
//            printf("%d ", (int) output[i * nchans + j]);
//        printf("\n");
//    }
}
