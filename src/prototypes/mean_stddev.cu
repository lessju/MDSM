#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include "time.h"

#define NUM_THREADS 256
#define NUM_BLOCKS  512

int nsamp = 65536 * 1024 * 4;

__global__ void mean_stddev(float *input, float2 *stddev, const int nsamp)
{
    // Declare shared memory to store temporary mean and stddev
    __shared__ float local_mean[NUM_THREADS];
    __shared__ float local_stddev[NUM_THREADS];

    // Initialise shared memory
    local_stddev[threadIdx.x] = 0;
    local_mean[threadIdx.x]   = 0;

    // Synchronise threads
    __syncthreads();

    // Loop over samples
    for(unsigned s = threadIdx.x + blockIdx.x * blockDim.x; 
                 s < nsamp; 
                 s += blockDim.x * gridDim.x)
    {
        float val = input[s];
        local_stddev[threadIdx.x] += (val * val);
        local_mean[threadIdx.x]   += val; 
    }

    // Synchronise threads
    __syncthreads();

    // Use reduction to calculate block mean and stddev
	for (unsigned i = NUM_THREADS / 2; i >= 1; i /= 2)
	{
		if (threadIdx.x < i)
		{
            local_stddev[threadIdx.x] += local_stddev[threadIdx.x + i];
            local_mean[threadIdx.x]   += local_mean[threadIdx.x + i];
		}
		
		__syncthreads();
	}

    // Finally, return temporary standard deviation value
    if (threadIdx.x == 0)
    {
        float2 vals = { local_mean[0], local_stddev[0] };
        stddev[blockIdx.x] = vals;
    }
}

int main(int argc, char *argv[])
{
	float  *input, *d_input;
    float2 *stddev, *d_stddev;
	int i, j;

	// Initialise
	cudaSetDevice(1);
	cudaThreadSetCacheConfig(cudaFuncCachePreferShared);
	
	// Allocate and generate buffers
    cudaMallocHost((void **) &input, nsamp * sizeof(float));
    cudaMallocHost((void **) &stddev, NUM_BLOCKS * sizeof(float2));

    for(j = 0; j < nsamp; j++)
        input[j] = rand() % 50;

	printf("Number of samples: %d\n", nsamp);

	cudaEvent_t event_start, event_stop;
	float timestamp;
	cudaEventCreate(&event_start); 
	cudaEventCreate(&event_stop); 

	// Allocate GPU memory and copy data
    cudaEventRecord(event_start, 0);
	cudaMalloc((void **) &d_input, nsamp * sizeof(float) );
	cudaMalloc((void **) &d_stddev, NUM_BLOCKS * sizeof(float2) );
    cudaMemcpy(d_input, input, nsamp * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Copied data to GPU in: %lf\n", timestamp);

	// Call kernel
	cudaEventRecord(event_start, 0);
    mean_stddev<<<NUM_BLOCKS, NUM_THREADS>>>(d_input, d_stddev, nsamp);
	cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Calculated mean and standard deviation: %lf\n", timestamp);

	// Get output 
	cudaMemcpy(stddev, d_stddev, NUM_BLOCKS * sizeof(float2), cudaMemcpyDeviceToHost);
    printf("Copied back results\n");

    // Calculate final mean and standard deviation
    float mean = 0, std = 0;
    for(i = 0; i < NUM_BLOCKS; i++)
    {
        mean += stddev[i].x;
        std  += stddev[i].y;
    }
    mean /= nsamp;
    std   = sqrt((std / nsamp)- mean * mean);

    printf("[GPU] Mean: %f, stddev: %f\n", mean, std);

    // Calculate mean and standard deviation on CPU
    double temp_mean = 0, temp_std = 0;
    for(j = 0; j < nsamp; j++)
    {
        temp_mean += input[j];
        temp_std += input[j] * input[j];
    }
    temp_mean /= nsamp;
    temp_std = sqrt((temp_std / nsamp) - temp_mean * temp_mean);
    printf("[CPU] Mean: %f, stddev: %f\n", temp_mean, temp_std);
}
