#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "time.h"
#include "string.h"

#ifndef DEDISP_THREADS
    #define DEDISP_THREADS  128  
#endif

#ifndef DEDISP_DMS
    #define DEDISP_DMS      32   
#endif

#define NUMREG 8
#define DIVINT 4
#define DIVINDM 32

float fch1 = 418, foff = -0.01953125, tsamp = 0.0000512, dmstep = 0.1, startdm = 0;
int nchans = 1024, nsamp = 131072, tdms = 4096, compare = 0;

// ======================== CUDA HELPER FUNCTIONS ==========================

// Error checking function
#define CUDA_ERROR_CHECK
#define CudaSafeCall( err ) _cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    _cudaCheckError( __FILE__, __LINE__ )

inline void _cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void _cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }   
#endif

    return; 
}

// ======================= Wes' Dedispersion Loop =============================
__global__ void cache_dedispersion(float *output, float *input, float *dm_shifts, 
                                   const int nsamp, const int nchans, const float mstartdm, 
                                   const float mdmstep, const int maxshift)
{
	int   shift;	
	float local_kernel_t[NUMREG];

	int t  = blockIdx.x * NUMREG * blockDim.x  + threadIdx.x;
	
	// Initialise the time accumulators
	for(int i = 0; i < NUMREG; i++) local_kernel_t[i] = 0.0f;

	float shift_temp = mstartdm + ((blockIdx.y * blockDim.y + threadIdx.y) * mdmstep);
	
	// Loop over the frequency channels.
    for(int c = 0; c < nchans; c++) 
    {
		// Calculate the initial shift for this given frequency
		// channel (c) at the current despersion measure (dm) 
		// ** dm is constant for this thread!!**
		shift = (c * (nsamp + maxshift) + t) + (dm_shifts[c] * shift_temp);
		
        #pragma unroll
		for(int i = 0; i < NUMREG; i++) {
			local_kernel_t[i] += input[shift + (i * DIVINT) ];
		}
	}

	// Write the accumulators to the output array. 
    #pragma unroll
	for(int i = 0; i < NUMREG; i++) {
		output[((blockIdx.y * DIVINDM) + threadIdx.y)* nsamp + (i * DIVINT) + (NUMREG * DIVINT * blockIdx.x) + threadIdx.x] = local_kernel_t[i];
	}
}

// ======================= Optimised Dedispersion Loop 1 =======================
__global__ void dedisperse_loop1(const float* __restrict__ input, float* __restrict__ output, 
							     const int* __restrict__ all_delays, const unsigned nchans, 
                                 const unsigned nsamp, const int maxshift, const int tdms)
{
	// Shared memory buffer to store channel vector
	extern __shared__ float vector[];

	// Each thread will process a number of DM values associated with one time sample
	register float accumulators[DEDISP_DMS];

	// Initialise shared memory store for dispersion delays
	__shared__ int delays[DEDISP_DMS];

	// Initialise accumulators
	for(unsigned d = 0; d < DEDISP_DMS; d++) accumulators[d] = 0;

	// Loop over all frequency channels
	for(unsigned c = 0; c < nchans; c++)
	{
		// Synchronise threads before updating dispersion delays
		__syncthreads();

		// Load all the shifts associated with this threadblock DM-range for the current channel
		int inshift = all_delays[c * tdms + blockIdx.y * DEDISP_DMS];
		if (threadIdx.x < DEDISP_DMS)
			delays[threadIdx.x] = all_delays[c * tdms + blockIdx.y * DEDISP_DMS + threadIdx.x] - inshift;
		
		// Synchronise threads
		__syncthreads();

		// We'll need to load the channel vector (which will be larger than threadDim
		// due to dispersion
		for(unsigned s = threadIdx.x; 
					 s < blockDim.x + delays[DEDISP_DMS - 1]; 
					 s += blockDim.x)
			vector[s] = input[(maxshift + nsamp) * c + blockIdx.x * blockDim.x + inshift + s];

		// Synchronise threads
		__syncthreads();

		// Loop over DM values associated with current threadblock and update accumulators
		// Manual unlooping of four to overlap shared memory requests
		#pragma unroll
		for(int d = 0; d < DEDISP_DMS; d ++)
	            accumulators[d]  += vector[threadIdx.x + delays[d]];
	}

	// All done, store result to global memory
    #pragma unroll
	for(unsigned d = 0; d < DEDISP_DMS; d++)
		output[(blockIdx.y * DEDISP_DMS + d) * nsamp + blockIdx.x * blockDim.x + threadIdx.x] = accumulators[d];
}

// ======================= Main Program =======================

// Process command-line parameters
void process_arguments(int argc, char *argv[])
{
    int i = 1;
    
    while(i < argc) {
       if (!strcmp(argv[i], "-nchans"))
           nchans = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-nsamp"))
           nsamp = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-dmstep"))
           dmstep = atof(argv[++i]);
       else if (!strcmp(argv[i], "-startdm"))
           startdm = atof(argv[++i]);
       else if (!strcmp(argv[i], "-tdms"))
           tdms = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-tsamp"))
           tsamp = atof(argv[++i]);
       else if (!strcmp(argv[i], "-foff"))
           foff = -atof(argv[++i]);
       else if (!strcmp(argv[i], "-fch1"))
           fch1 = atof(argv[++i]);
       else if (!strcmp(argv[i], "-compare"))
           compare = 1;
       i++;
    }
}

// Fill buffer with data (blocking call)
void generate_data(float* buffer, int nsamp, int nchans)
{
    for(int i = 0; i < nsamp * nchans; i++)
        buffer[i] = 0.1;
}

// DM delay calculation
float dmdelay(float f1, float f2)
{
  return(4148.741601 * ((1.0 / f1 / f1) - (1.0 / f2 / f2)));
}

int main(int argc, char *argv[])
{
   float *input, *output, *d_input, *d_output, *d_delays;
   int maxshift;

   process_arguments(argc, argv);

    // Calculate temporary DM-shifts
    float *dmshifts = (float *) malloc(nchans * sizeof(float));
    for (unsigned i = 0; i < nchans; i++)
          dmshifts[i] = dmdelay(fch1 + (foff * i), fch1) / tsamp;

    // Calculate maxshift
    maxshift = ceil(dmshifts[nchans - 1] * (startdm + dmstep * tdms));

    // Allocate and initialise arrays
    input = (float *) malloc( (nsamp + maxshift) * nchans * sizeof(float));
    memset(input, 0, (nsamp + maxshift) * nchans * sizeof(float));
    output = (float *) malloc( nsamp * tdms * sizeof(float));

    // Initialise CUDA stuff
    CudaSafeCall(cudaSetDevice(1));
	CudaSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    cudaEvent_t event_start, event_stop;
    float timestamp, kernelTime;

    cudaEventCreate(&event_start); 
    cudaEventCreate(&event_stop);

    printf("dedisp_threads: %d, dedisp_dm: %d\n", DEDISP_THREADS, DEDISP_DMS);
    printf("nsamp: %d, nchans: %d, tsamp: %f, startdm: %f, dmstep: %f, tdms: %d, fch1: %f, foff: %f, maxshift: %d\n",
           nsamp, nchans, tsamp, startdm, dmstep, tdms, fch1, foff, maxshift);

    printf("Memory requirements: Input: %.2f MB, Output: %.2f MB \n", nchans * (nsamp + maxshift) * sizeof(float) / (1024.0 * 1024),
                                                                      tdms * nsamp * sizeof(float) / (1024.0 * 1024.0));

    // Allocate CUDA memory and copy dmshifts
    CudaSafeCall(cudaMalloc((void **) &d_input, (nsamp + maxshift) * nchans * sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_output, nsamp * tdms * sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_delays, nchans * sizeof(float)));
    CudaSafeCall(cudaMemset(d_output, 0, nsamp * tdms * sizeof(float)));
	memset(output, 0, nsamp * tdms * sizeof(float));

    time_t start = time(NULL);

    // Copy input to GPU
    cudaEventRecord(event_start, 0);
    CudaSafeCall(cudaMemcpy(d_input, input, (nsamp + maxshift) * nchans * sizeof(float), cudaMemcpyHostToDevice) );    
    CudaSafeCall(cudaMemcpy(d_delays, dmshifts, nchans * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied to GPU in: %lf\n", timestamp);

    // Dedisperse using Wes' kernel
    if (compare)
    {
        int num_reg         = NUMREG;
        int divisions_in_t  = DIVINT;
        int divisions_in_dm = DIVINDM;
        int num_blocks_t    = nsamp / (divisions_in_t * num_reg);
        int num_blocks_dm   = tdms / divisions_in_dm;

        dim3 threads_per_block(divisions_in_t, divisions_in_dm);
        dim3 num_blocks(num_blocks_t,num_blocks_dm); 

        cudaEventRecord(event_start, 0);	
        cache_dedispersion<<< num_blocks, threads_per_block >>>
                          (d_output, d_input, d_delays, nsamp, nchans, startdm, dmstep, maxshift);
        cudaEventRecord(event_stop, 0);
        cudaEventSynchronize(event_stop);
        cudaEventElapsedTime(&timestamp, event_start, event_stop);
        printf("Performed Cache Dedispersion %lf\n", timestamp);
	
    }   
	// Cacluate the extra shared memory required to store shifts
	unsigned shift = round(dmshifts[nchans - 1] * (startdm + tdms * dmstep)) - 
                     round(dmshifts[nchans - 1] * (startdm + (tdms - DEDISP_DMS) * dmstep));

	// Pre-compute channel and DM specific shifts beforehand on CPU
	// This only needs to be computed once for the entire execution
	int *all_shifts = (int *) malloc(nchans * tdms * sizeof(int));
	for(unsigned c = 0; c < nchans; c++)
		for (unsigned d = 0; d < tdms; d++)
			all_shifts[c * tdms + d] = (int) (dmshifts[c] * (startdm + (d * dmstep)));

	int *d_all_shifts;
	CudaSafeCall(cudaMalloc((void **) &d_all_shifts, nchans * tdms * sizeof(int)));
	CudaSafeCall(cudaMemcpy(d_all_shifts, all_shifts, nchans * tdms * sizeof(int), cudaMemcpyHostToDevice) );  

	dim3 gridDim(ceil(nsamp / (1.0 * DEDISP_THREADS)), ceil(tdms / (1.0 * DEDISP_DMS)));  
    cudaEventRecord(event_start, 0);
    dedisperse_loop1 <<< gridDim, DEDISP_THREADS, (DEDISP_THREADS + shift) * sizeof(float) >>> 
       (d_input, d_output, d_all_shifts, nchans, nsamp, maxshift, tdms);

    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Perform Shared Memory Dedispersion [v1] in: %lf\n", timestamp);
    kernelTime = timestamp;

    // Copy output from GPU
    cudaEventRecord(event_start, 0);
    CudaSafeCall(cudaMemcpy(output, d_output, nsamp * tdms * sizeof(float), cudaMemcpyDeviceToHost) );    
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied from GPU in: %lf\n", timestamp);

    printf("Performance: %lf Gflops\n", (nchans * tdms * 6) * (nsamp * 1.0 / kernelTime / 1.0e6));
}

