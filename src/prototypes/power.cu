#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <sys/time.h>
#include "time.h"
#include <gsl/gsl_multifit.h>
#include <math.h>
#include "file_handler.h"
 
#define BANDPASS_THREADS 512
#define ANSWER(i,j) i*i + (j%8192)*(j%8192)
#define SUM(a,b,c) a + b + c

int nsamp = 262144, nchans = 512;

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


// ==========================================================================

// Compute power from input complex values
// A[N] = A[N].x * A[N].x + A[N].y * A[N].y
// Performed in place (data will still consume 32-bits in GPU memory)
__global__ void power(float *data, unsigned nchans, unsigned shift, unsigned samples, unsigned total)
{
    for(unsigned c = 0; c < nchans; c++)
        for(unsigned s = blockIdx.x * blockDim.x + threadIdx.x; 
                     s < samples;
                     s += gridDim.x * blockDim.x)
        {
            short2 value = *((short2 *) &(data[c * total + shift + s]));
            data[c * total + shift + s] = value.x * value.x + value.y * value.y;
        }
}


// Main function
int main(int argc, char *argv[])
{
	cudaEvent_t event_start, event_stop;
	float timestamp;
	cudaEventCreate(&event_start); 
	cudaEventCreate(&event_stop); 

    // Allocate and initialise CPU and GPU memory for data and bandpass
    float *buffer, *d_buffer;
    CudaSafeCall(cudaMallocHost((void **) &buffer, nchans * nsamp * sizeof(float), cudaHostAllocPortable));

    // Generate fake data
    short2 *data = (short2 *) buffer;
    for(unsigned i = 0; i < nchans; i++)
        for(unsigned j = 0; j < nsamp; j++)
            {   
                data[i*nsamp+j].x = i; 
                data[i*nsamp+j].y = j % 8192; 
            }

    cudaMalloc((void **) &d_buffer, nchans * nsamp * sizeof(float));

    // Copy input buffer to GPU memory
    cudaEventRecord(event_start, 0);
    cudaMemcpy(d_buffer, buffer, nchans * nsamp * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Copied data to GPU in: %lf\n", timestamp);

    // Calculate power
    cudaEventRecord(event_start, 0);
    dim3 blocks;
    power<<< ceil(nsamp / (1.0 * BANDPASS_THREADS)), BANDPASS_THREADS>>>(d_buffer, nchans, 0, nsamp, nsamp);
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Computed voltage power in : %lf\n", timestamp);

    // Copy result back to CPU memory
    cudaEventRecord(event_start, 0);

    CudaSafeCall(cudaMemcpy(buffer, d_buffer, nchans * nsamp * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Copied results back to CPU memory in : %lf\n", timestamp);

    // Check results
    for(unsigned i = 0; i < nchans; i++)
        for(unsigned j = 0; j < nsamp; j++)
            if (buffer[i*nsamp+j] != ANSWER(i,j))
                { printf("[%d.%d] %f != %f\n", j, i, buffer[i*nsamp+j], ANSWER(i,j)); exit(0);}
}

    
