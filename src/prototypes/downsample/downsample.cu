#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "time.h"
#include "string.h"

float tsamp = 0.0000512, req_dtime = 0.001;
int nchans = 1024, nsamp = 32768, nbeams = 16, factor = 64;

#define ANTS 32

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

// ======================= Downfactor Kernel =============================
// Downfactor generated beam down to the required sampling time
__global__ void downsample(float *input, float *output, unsigned nsamp, unsigned nchans, unsigned nbeams, unsigned factor)
{
    // Each thread block processes one channel/beam vector

    // Loop over all time samples
    for(unsigned s = 0;
                 s < nsamp / (blockDim.x * factor);
                 s++)
    {
        // Index for thread block starting point
        unsigned index = (blockIdx.y * nchans + blockIdx.x) * nsamp + s * blockDim.x * factor;

        // Perform local downsampling and store to local accumulator
        float value = 0;
        for(unsigned i = 0; i < factor; i++)
            value += input[index + threadIdx.x * factor + i];

        // Output needs to be transposed in memory
        output[(s * blockDim.x + threadIdx.x) * nchans * nbeams + blockIdx.x * nbeams + blockIdx.y] = value;
    }
}

// Downfactor generated beam down to the required sampling time
__global__ void downsample_reduce(float *input, float *output, unsigned nsamp, unsigned nchans, unsigned nbeams, unsigned factor)
{
    // Each thread block processes one channel/beam vector
    extern __shared__ float vector[];

    // Loop over all time samples
    for(unsigned s = blockIdx.x;
                 s < nsamp / factor;
                 s += gridDim.x)
    {
        // blockDim.x == factor, so each thread loads one value
        vector[threadIdx.x] = input[(blockIdx.z * nchans + blockIdx.y) * nsamp + s * factor + threadIdx.x];

        // Synchronise threads
        __syncthreads();

        // Use reduction to calculate block mean and stddev
	    for (unsigned i = blockDim.x / 2; i >= 1; i /= 2)
	    {
		    if (threadIdx.x < i)
                vector[threadIdx.x] += vector[threadIdx.x + i];
		
		    __syncthreads();
	    }

        // Store to output (transposed)
        if (threadIdx.x == 0)
            output[s * nchans * nbeams + blockIdx.y * nbeams + blockIdx.z] = vector[0];

        // Synchronise threads
        __syncthreads();
    
    }
}

// Downfactor generated beam down to the required sampling time using atomics
__global__ void downsample_atomics(float *input, float *output, unsigned nsamp, unsigned nchans, unsigned nbeams, unsigned factor)
{
    // Each thread block processes one channel/beam vector
    extern __shared__ float vector[];

    // Loop over all time samples
    for(unsigned s = blockIdx.x;
                 s < nsamp / factor;
                 s += gridDim.x)
    {
        // Load input value
        float local_value = input[(blockIdx.z * nchans + blockIdx.y) * nsamp + s * factor + threadIdx.x];

        // Use kepler atomics to perform partial reduction
        local_value += __shfl_down(local_value, 1, 2);
        local_value += __shfl_down(local_value, 2, 4);
        local_value += __shfl_down(local_value, 4, 8);
        local_value += __shfl_down(local_value, 8, 16);
        local_value += __shfl_down(local_value, 16, 32);

        // Synchronise thread to finalise first part of partial reduction
        __syncthreads();

        // Store required value to shared memory
        if (threadIdx.x % 32 == 0)
            vector[threadIdx.x / 32] = local_value;

        // Synchronise thread
        __syncthreads();

        // Perform second part of reduction
        for (unsigned i = factor / 64; i >= 1; i /= 2)
	    {
		    if (threadIdx.x < i)
                vector[threadIdx.x] += vector[threadIdx.x + i];
		
		    __syncthreads();
	    }

        // Store to output (transposed)
        if (threadIdx.x == 0)
            output[s * nchans * nbeams + blockIdx.y * nbeams + blockIdx.z] = vector[0];

        // Synchronise threads
        __syncthreads();
    
    }
}

// ======================= Rearrange Kernel =============================
// Rearrange medicina antenna data to match beamformer required input 
// Heap size is 1024 channels, 128 samples, 32 antennas (8 bytes each)
// Threadblock size is 128 samples
#define HEAP 128

__global__ void rearrange_medicina(unsigned char *input, unsigned char *output, unsigned nsamp, unsigned nchans)
{
    // Each grid row processes a separate channel
    // Each grid column processes a separate heap
    for(unsigned h = blockIdx.x;
                 h < nsamp / blockDim.x;
                 h += gridDim.x)
    {
        unsigned int indexIn = blockIdx.y * nsamp * ANTS + h * HEAP;
        unsigned int indexOu = blockIdx.y * nsamp + ANTS + h * HEAP * ANTS;
     
        // Thread ID acts as pointer to required sample
        for(unsigned a = 0; a < ANTS * 0.5; a++)
        {
            output[indexOu + threadIdx.x * ANTS + a * 2 + 1] = input[indexIn + a * HEAP * 2 + threadIdx.x * 2 + 1];
            output[indexOu + threadIdx.x * ANTS + a * 2    ] = input[indexIn + a * HEAP * 2 + threadIdx.x * 2];
        }
    }
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
       else if (!strcmp(argv[i], "-tsamp"))
           tsamp = atof(argv[++i]);
       else if (!strcmp(argv[i], "-nbeams"))
           nbeams = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-factor"))
           factor = atoi(argv[++i]);
       i++;
    }
}

int main(int argc, char *argv[])
{
    float *input, *d_input, *output, *d_output;

    process_arguments(argc, argv);

    printf("nsamp: %d, nchans: %d, nbeams: %d, tsamp: %f, factor: %d\n", nsamp, nchans, nbeams, tsamp, factor);
    printf("Memory requirements: Input: %.2f MB, Output: %.2f \n", nchans * nbeams * nsamp * sizeof(float) / (1024.0 * 1024),
                                                                   nchans * nbeams * (nsamp / factor) * sizeof(float) / (1024.0 * 1024));

    // Allocate and initialise arrays
    input  = (float *) malloc( nsamp * nchans * nbeams * sizeof(float));
    output = (float *) malloc( nsamp * nchans * nbeams * sizeof(float) / factor);
    memset(input,  0, nsamp * nchans * nbeams * sizeof(float));
    memset(output, 0, nsamp * nchans * nbeams * sizeof(float) / factor);

    for(unsigned i = 0; i < nbeams; i++)
        for(unsigned j = 0; j < nchans; j++)
            for(unsigned k = 0; k < nsamp/factor; k++)
                for(unsigned l = 0; l < factor; l++)
                    input[i * nchans * nsamp + j * nsamp + k * factor + l] = k;

    // Initialise CUDA stuff
    CudaSafeCall(cudaSetDevice(0));
	CudaSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    cudaEvent_t event_start, event_stop;
    float timestamp, kernelTime;

    cudaEventCreate(&event_start); 
    cudaEventCreate(&event_stop);

    // Allocate GPU memory
    CudaSafeCall(cudaMalloc((void **) &d_input, nsamp * nchans * nbeams * sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_output, nsamp * nchans * nbeams * sizeof(float) / factor));
    CudaSafeCall(cudaMemset(d_output, 0, nsamp * nchans * nbeams * sizeof(float) / factor));

    time_t start = time(NULL);

    // Copy input to GPU
    cudaEventRecord(event_start, 0);
    CudaSafeCall(cudaMemcpy(d_input, input, nsamp * nbeams * nchans * sizeof(float), cudaMemcpyHostToDevice));    
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied to GPU in: %lf\n", timestamp); 

    if (factor <= 32)
    {
        unsigned num_threads = 128;
	    dim3 gridDim(nchans, nbeams);  
        cudaEventRecord(event_start, 0);
        downsample <<< gridDim, num_threads >>> (d_input, d_output, nsamp, nchans, nbeams, factor);
        cudaEventRecord(event_stop, 0);
        cudaEventSynchronize(event_stop);
        cudaEventElapsedTime(&timestamp, event_start, event_stop);
        printf("Perform downsampling in: %lf [%.2f GFLOPs] \n", timestamp, nsamp * nchans * nbeams / (timestamp * 1e-3) * 1e-9);
        kernelTime = timestamp;
    }

    // Launch shared memory version 
    else
    {
        unsigned num_threads = factor;
	    dim3 gridDim((nsamp / factor), nchans, nbeams);  
        cudaEventRecord(event_start, 0);
        downsample_atomics <<< gridDim, num_threads, num_threads * sizeof(float) / 32 >>> (d_input, d_output, nsamp, nchans, nbeams, factor);
        cudaEventRecord(event_stop, 0);
        cudaEventSynchronize(event_stop);
        cudaEventElapsedTime(&timestamp, event_start, event_stop);
        printf("Perform downsampling in: %lf [%.2f GFLOPs] \n", timestamp, nsamp * nchans * nbeams / (timestamp * 1e-3) * 1e-9);
        kernelTime = timestamp;
    }

    // Copy output from GPU
    cudaEventRecord(event_start, 0);
    CudaSafeCall(cudaMemcpy(output, d_output, nsamp * nbeams * nchans * sizeof(float) / factor, cudaMemcpyDeviceToHost) );    
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied from GPU in: %lf\n", timestamp);

    for(unsigned i = 0; i < nbeams; i++)
        for(unsigned j = 0; j < nchans; j++)
            for(unsigned k = 0; k < nsamp / factor; k++)
                if ((int) output[k * nchans * nbeams + j * nbeams + i] != k * factor)
                {
                    printf("[%d,%d,%d] %f != %d\n", i,j,k, output[k * nchans * nbeams + j * nbeams + i], k * factor);
                    exit(0);
                }        
}

