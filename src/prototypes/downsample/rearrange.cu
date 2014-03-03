#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "time.h"
#include "string.h"

int nchans = 1024, nsamp = 32768;

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
                 h < nsamp / HEAP;
                 h += gridDim.x)
    {
        unsigned int index = blockIdx.y * nsamp * ANTS + h * HEAP * ANTS;
     
        // Thread ID acts as pointer to required sample
        for(unsigned a = 0; a < ANTS * 0.5; a++)
        {
            output[index + threadIdx.x * ANTS + a * 2 + 1] = input[index + a * HEAP * 2 + threadIdx.x * 2 + 1];
            output[index + threadIdx.x * ANTS + a * 2    ] = input[index + a * HEAP * 2 + threadIdx.x * 2];
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
       i++;
    }
}

int main(int argc, char *argv[])
{
    unsigned char *input, *d_input, *output, *d_output;

    process_arguments(argc, argv);

    printf("nsamp: %d, nchans: %d\n", nsamp, nchans);
    printf("Memory requirements: Input: %.2f MB, Output: %.2f \n", nchans * ANTS * nsamp * sizeof(unsigned char) / (1024.0 * 1024),
                                                                   nchans * ANTS * nsamp * sizeof(unsigned char) / (1024.0 * 1024));

    // Allocate and initialise arrays
    input  = (unsigned char *) malloc( nsamp * nchans * ANTS * sizeof(unsigned char));
    output = (unsigned char *) malloc( nsamp * nchans * ANTS * sizeof(unsigned char));
    memset(input,  0, nsamp * nchans * ANTS * sizeof(unsigned char));
    memset(output, 0, nsamp * nchans * ANTS * sizeof(unsigned char));

    for (unsigned c = 0; c < nchans; c++)
        for(unsigned s = 0; s < nsamp / HEAP; s++)
            for(unsigned a = 0; a < ANTS * 0.5; a++)
                for(unsigned t = 0; t < HEAP; t++)
                {
                    input[c * ANTS * nsamp + s * ANTS * HEAP + a * HEAP * 2 + t * 2 + 1] = a * 2 + 1;
                    input[c * ANTS * nsamp + s * ANTS * HEAP + a * HEAP * 2 + t * 2] = a * 2;
                }
        

    // Initialise CUDA stuff
    CudaSafeCall(cudaSetDevice(0));
	CudaSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    cudaEvent_t event_start, event_stop;
    float timestamp, kernelTime;

    cudaEventCreate(&event_start); 
    cudaEventCreate(&event_stop);

    // Allocate GPU memory
    CudaSafeCall(cudaMalloc((void **) &d_input, nsamp * nchans * ANTS * sizeof(unsigned char)));
    CudaSafeCall(cudaMalloc((void **) &d_output, nsamp * nchans * ANTS * sizeof(unsigned char)));

    time_t start = time(NULL);

    // Copy input to GPU
    cudaEventRecord(event_start, 0);
    CudaSafeCall(cudaMemcpy(d_input, input, nsamp * ANTS * nchans * sizeof(unsigned char), cudaMemcpyHostToDevice));    
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied to GPU in: %lf\n", timestamp); 

    // Launch shared memory version 
    cudaEventRecord(event_start, 0);
	dim3 gridDim(nsamp / HEAP, nchans);  
    rearrange_medicina <<< gridDim, HEAP >>> (d_input, d_output, nsamp, nchans);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Perform rearrangement in: %lf [%.2f GFLOPs] \n", timestamp, nsamp * nchans * ANTS / (timestamp * 1e-3) * 1e-9);
    kernelTime = timestamp;

    // Copy output from GPU
    cudaEventRecord(event_start, 0);
    CudaSafeCall(cudaMemcpy(output, d_output, nsamp * ANTS * nchans * sizeof(unsigned char), cudaMemcpyDeviceToHost) );    
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied from GPU in: %lf\n", timestamp);

    for(unsigned c = 0; c < nchans; c++)
        for(unsigned s = 0; s < nsamp; s++)
            for(unsigned a = 0; a < ANTS; a++)
                if ( ((int) output[c*nsamp*ANTS + s*ANTS+a]) != a)
                {
                    printf("[%d,%d,%d] %d != %d\n", c,s,a, (int) output[c*nsamp*ANTS + s*ANTS+a], a);
//                    exit(0);
                }        
}

