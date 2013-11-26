#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include "time.h"

#define NUM_THREADS 128
#define MEDIAN_WIDTH 7

int nsamp = 262144, ndms = 1024;

__global__ __device__ void median_filter(float *input, const int nsamp)
{
    // Declare shared memory array to hold local kernel samples
    // Should be (blockDim.x+width floats)
    __shared__ float local_kernel[NUM_THREADS + MEDIAN_WIDTH - 1];

    // Value associated with thread
    unsigned samp = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned index = blockIdx.y * nsamp + samp;
    unsigned wing  = MEDIAN_WIDTH / 2;

    // Load sample associated with thread into shared memory
    local_kernel[threadIdx.x + wing] = input[index];

    __syncthreads();

    // Load kernel wings into shared memory, handling boundary conditions
    // (for first and last wing elements in time series)
    if (samp >= wing && samp < nsamp - wing)
    {
        // Not in boundary, choose threads at the edge and load wings
        if (threadIdx.x < wing)   // Load wing element at the beginning of the kernel
            local_kernel[threadIdx.x] = input[blockIdx.y * nsamp + blockIdx.x * blockDim.x - (wing - threadIdx.x)];
        else if (threadIdx.x >= blockDim.x - wing)  // Load wing elements at the end of the kernel
            local_kernel[threadIdx.x + MEDIAN_WIDTH - 1] = input[index + wing];
    }

    // Handle boundary conditions (ignore end of buffer for now)
    else if (samp < wing && threadIdx.x < wing + 1)   
        // Dealing with the first items in the input array
        local_kernel[threadIdx.x] = local_kernel[wing];
    else if (samp > nsamp - wing && threadIdx.x == blockDim.x / 2)
        // Dealing with last items in the input array
        for(unsigned i = 0; i < wing; i++)
            local_kernel[NUM_THREADS + wing + i] = local_kernel[nsamp - 1];

    // Synchronise all threads and start processing
    __syncthreads();

    // Load value to local registers median using "moving window" in shared memory 
    // to avoid bank conflicts
    float median[MEDIAN_WIDTH];
    for(unsigned i = 0; i < MEDIAN_WIDTH; i++)
        median[i] = local_kernel[threadIdx.x + i];

    // Perform partial-sorting on median array
    for(unsigned i = 0; i < wing + 1; i++)    
        for(unsigned j = i; j < MEDIAN_WIDTH; j++)
            if (median[j] < median[i])
                { float tmp = median[i]; median[i] = median[j]; median[j] = tmp; }

    // We have our median, store to global memory
    input[index] = median[wing];
}

int main(int argc, char *argv[])
{
	float *input, *d_input;
	int i, j;

	// Initialise
	cudaSetDevice(0);
	cudaThreadSetCacheConfig(cudaFuncCachePreferShared);
	
	// Allocate and generate buffers
    cudaMallocHost((void **) &input, nsamp * ndms * sizeof(float));
//    for(i = 0; i < ndms; i++)
//        for(j = 0; j < nsamp; j++)
//            input[i * nsamp + j] = j; 
    // Load in data
//    FILE *fp = fopen("median-test.dat", "r");
//    fread(input, sizeof(float), nsamp * ndms, fp);
//    fclose(fp);

	printf("nsamp: %d, ndms: %d\n", nsamp, ndms);

	cudaEvent_t event_start, event_stop;
	float timestamp;
	cudaEventCreate(&event_start); 
	cudaEventCreate(&event_stop); 

	// Allocate GPU memory and copy data
    cudaEventRecord(event_start, 0);
	cudaMalloc((void **) &d_input, nsamp * ndms * sizeof(float) );
	cudaMemcpy(d_input, input, nsamp * ndms * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Copied data to GPU in: %lf\n", timestamp);

	// Call kernel
	cudaEventRecord(event_start, 0);
//    printf("%d %d %d\n", NUM_BLOCKS, ndms, NUM_THREADS);
    median_filter<<<dim3(nsamp / NUM_THREADS, ndms), NUM_THREADS>>>(d_input, nsamp);
	cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Performed median filtering in: %lf\n", timestamp);

	// Get output 
	cudaMemcpy(input, d_input, ndms * nsamp * sizeof(float), cudaMemcpyDeviceToHost);

    // Write result
//    fp = fopen("median-test-gpu.dat", "wb");
//    fwrite(input, sizeof(float), nsamp * ndms, fp);
//    fclose(fp);


//    for(i = 0; i < ndms; i++)
//        for(j = 0; j < nsamp - 3; j++)
//            if (abs(input[i * nsamp + j] - j) > 2 && input[i*nsamp+j]!=0)
//                { printf("%d,%d. %f\n", i, j, input[i * nsamp + j]);  }
}
