#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "time.h"

#define NUM_THREADS 256

int nsamp = 65536, ndms = 1024, detrendLen = 32768;

// --------------------------- Detrend and Normalisation kernel ----------------------------
__global__ void detrend_normalise(float *input, int detrendLen)
{	
	// Store temporary least-square fit values
	__shared__ float3 shared[NUM_THREADS];

	// First pass, each thread computes its part of the buffer
	{
		float sy = 0, sxy = 0, sxx = 0;
		for (unsigned i = threadIdx.x; i < detrendLen; i += blockDim.x)
		{
			float x = - detrendLen * 0.5 + 0.5 + i;
			int index = blockIdx.y * gridDim.x * detrendLen + 
						blockIdx.x * detrendLen + i;
			float y = input[index];

			sy += y;
			sxy += x * y;
			sxx += x * x;
		}

		// Initialise shared memory
		shared[threadIdx.x].x = sy;
		shared[threadIdx.x].y = sxy;
		shared[threadIdx.x].z = sxx;
	}

	__syncthreads();

	// Perform the rest of the computation through reduction
	for (unsigned i = NUM_THREADS / 2; i >= 1; i /= 2)
	{
		if (threadIdx.x < i)
		{
			shared[threadIdx.x].x += shared[threadIdx.x + i].x;
			shared[threadIdx.x].y += shared[threadIdx.x + i].y;
			shared[threadIdx.x].z += shared[threadIdx.x + i].z;
		}
		
		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		shared[0].y /= shared[0].z;
		shared[0].x /= detrendLen;
	}

	__syncthreads();
	
	// Detrend and compute partial standard deviation
	{
		float a = shared[0].x;
		float b = shared[0].y;
		float stddev = 0;

		for (unsigned i = threadIdx.x; i < detrendLen ; i += blockDim.x)
		{
			float x = - detrendLen / 2.0 + 0.5 + i;
			int index = blockIdx.y * gridDim.x * detrendLen + 
						blockIdx.x * detrendLen + i;
			float val = input[index] - (a + b * x);
			input[index] = val;
			stddev += val * val;
		}

		shared[threadIdx.x].z = stddev;
	}

	__syncthreads();

	// Compute the full standard deviation through reduction
	for (unsigned i = NUM_THREADS / 2; i >= 1; i /= 2)
		if (threadIdx.x < i)
			shared[threadIdx.x].z += shared[threadIdx.x + i].z;

	__syncthreads();

	if (threadIdx.x == 0)
		shared[0].z = sqrt(shared[0].z / detrendLen);

	__syncthreads();

	// Normalise Data
	float stddev = shared[0].z;

	for (unsigned i = threadIdx.x; i < detrendLen ; i += blockDim.x)
		input[blockIdx.y * gridDim.x * detrendLen + 
			  blockIdx.x * detrendLen + i] /= stddev;
}

// --------------------------- Main processing  ----------------------------


int main(int argc, char *argv[])
{
	float *input, *d_input;
	int i, j;
	
	// Allocate and generate buffers
	input = (float *) malloc(nsamp * ndms * sizeof(float));

	srand ( time(NULL) );
	for(i = 0; i < ndms; i++)
		for(j = 0; j < nsamp; j++)
			input[i * nsamp + j] = ((float)rand() / (float)RAND_MAX) + j * 0.001 + i * j * 0.001;

	printf("nsamp: %d, ndms: %d\n", nsamp, ndms);

//	FILE *fp = fopen("/home/lessju/Code/MDSM/release/pelican-mdsm/pipelines/output.dat", "rb");
//	fread(input, sizeof(float), nsamp, fp);
//	fclose(fp);
	
	// Initialise
	cudaSetDevice(0);
	cudaThreadSetCacheConfig(cudaFuncCachePreferL1);

	cudaEvent_t event_start, event_stop;
	float timestamp;
	cudaEventCreate(&event_start); 
	cudaEventCreate(&event_stop); 

	// Allocate GPU memory and copy data
	cudaMalloc((void **) &d_input, nsamp * ndms * sizeof(float) );
	cudaMemcpy(d_input, input, nsamp * ndms * sizeof(float), cudaMemcpyHostToDevice);

	// Call kernel
	cudaEventRecord(event_start, 0);
	detrend_normalise<<<dim3(nsamp / detrendLen, ndms), NUM_THREADS>>>(d_input, detrendLen);
	cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Performed detrending in: %lf\n", timestamp);

	// Get output 
	cudaMemcpy(input, d_input, ndms * nsamp * sizeof(float), cudaMemcpyDeviceToHost);

	FILE *fp = fopen("testDetrend.dat", "wb");
	fwrite(input, sizeof(float), nsamp * ndms, fp);
	fclose(fp);
}
