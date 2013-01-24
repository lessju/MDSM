#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <sys/time.h>
#include "time.h"
#include <math.h>

#define BEAMFORMER_THREADS 64
#define BEAMS 16
unsigned nchans = 1024, nants = 32, nsamp = 8192, nbeams = 16;

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


// ==========================================================================

__global__ void beamform_shared(char4 *input, float *output, unsigned nsamp,
                                unsigned nants, unsigned nchans, unsigned nbeams)
{
    __shared__ char4 shared[BEAMFORMER_THREADS * 16];
    __shared__ float beams[BEAMFORMER_THREADS * BEAMS];

    // Loop over all time samples
    for(unsigned time = blockIdx.x;
                 time < nsamp;
                 time += gridDim.x)
    {
        // Synchronise threads
        __syncthreads();

        // Loop over channels
        for(unsigned channel = threadIdx.x;
                     channel < nchans;
                     channel += blockDim.x)
        {
            // Load data to shared memory
            unsigned index = 16 * (time * nchans + (channel / blockDim.x) * blockDim.x );
            for(unsigned i = threadIdx.x;
                         i < blockDim.x * 16;
                         i += blockDim.x)
                shared[i] = input[index + i];

            // Initialise shared memory storing beams
            for(unsigned i = 0; i < nbeams; i++)
                beams[blockDim.x * i + threadIdx.x] = 0;
        
            // Synchronise threads
            __syncthreads();

            // Loop over all antennas and beamform
            for(unsigned antenna = 0;
            			 antenna < nants / 4;
            			 antenna ++)
            {
                char4 real = shared[threadIdx.x * 16 + (antenna + threadIdx.x) % 8];
				char4 imag = shared[threadIdx.x * 16 + 8 + (antenna + threadIdx.x) % 8];

                // Loop over all beams
				for(unsigned beam = 0; beam < nbeams; beam++)
				{
                    // Add four antennas at a time (to reduce shared memory overhead and increase arithmetic intensity)
                    float2 shift1 = {1, beam}, shift2 = {1, beam}, shift3 = {1, beam}, shift4 = {1, beam};
                    float temp1 = (shift1.x * real.w) * (shift1.x * real.w) + (shift1.y * imag.w) * (shift1.y * imag.w);
                    float temp2 = (shift2.x * real.x) * (shift2.x * real.x) + (shift2.y * imag.x) * (shift2.y * imag.x);
                    float temp3 = (shift3.x * real.y) * (shift3.x * real.y) + (shift3.y * imag.y) * (shift3.y * imag.y);
                    float temp4 = (shift4.x * real.z) * (shift4.x * real.z) + (shift4.y * imag.z) * (shift4.y * imag.z);
					beams[blockDim.x * beam + threadIdx.x] += temp1 + temp2 + temp3 + temp4;
				}
            }

            // Synchronise threads
            __syncthreads();

            // Save beam value to global memory
            for(unsigned beam = 0; beam < nbeams; beam++)
                output[beam * nsamp * nchans + channel * nsamp + time] = beams[blockDim.x * beam + threadIdx.x];//beams[blockDim.x * beam + threadIdx.x];
//                    output[blockIdx.x * blockDim.x * beam + threadIdx.x] = beams[beam];

            // Synchronise threads
            __syncthreads();
        }
    }
}

// ==========================================================================
int main(int agrc, char *argv[])
{
//    struct timeval start, end;
//    long mtime, seconds, useconds;

	cudaEvent_t event_start, event_stop;
	float timestamp;
	cudaEventCreate(&event_start);
	cudaEventCreate(&event_stop);

    printf("Memory requirements: Input: %.2f MB, Output: %.2f MB \n", nchans * nsamp * nants * sizeof(char2) / (1024.0 * 1024),
                                                             nchans * nsamp * nbeams * sizeof(float) / (1024.0 * 1024.0));
    printf("nsamp: %d, nchans: %d, nbeams: %d, nants: %d\n", nsamp, nchans, nbeams,  nants);

    // Allocate and initialise CPU and GPU memory for data
    // Data is stored in time/frequency/antenna order with antenna changing the fastest
    // Samples are stored in "block complex form", with 8-bit real/complex components packed
    // into 32 elements (32R,32I)...
    char *input_buffer, *d_input_buffer;
    CudaSafeCall(cudaMallocHost((void **) &input_buffer, nchans * nsamp * nants * sizeof(char2)));
    CudaSafeCall(cudaMalloc((void **) &d_input_buffer, nchans * nsamp * nants * sizeof(char2)));
    printf("Allocated input buffers\n");

    // We will be outputing beam to be processed by the transient detection pipeline,
    // whose required input data format is beam/channel/time, with time changing the faster,
    // and is in 32-bit single precision floating point
    float *d_output_buffer, *output_buffer;
    CudaSafeCall(cudaMallocHost((void **) &output_buffer, nchans * nsamp * nbeams * sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_output_buffer, nchans * nsamp * nbeams * sizeof(float)));
    printf("Allocated output buffers\n");

    // Generate fake data
    for(unsigned i = 0; i < nsamp; i++)
        for(unsigned j = 0; j < nchans; j++)
            for(unsigned k = 0; k < nants * 2; k++)
                input_buffer[i * nchans * nants * 2 + j * nants * 2 + k] = j;
//    memset(input_buffer, 1, nchans * nsamp * nants * sizeof(char2));
    printf("Generated fake data\n");

    // Copy input buffer to GPU
    cudaEventRecord(event_start, 0);
    CudaSafeCall(cudaMemcpy(d_input_buffer, input_buffer, nchans * nsamp * nants * sizeof(char2), cudaMemcpyHostToDevice));
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Copied data to GPU in: %lf\n", timestamp);

    // Run beamformer kernel
    cudaEventRecord(event_start, 0);
    beamform_shared<<< 4096, BEAMFORMER_THREADS >>>
            ((char4 *) d_input_buffer, d_output_buffer, nsamp, nants, nchans, nbeams);
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
    double kernel_time = timestamp;
	printf("Performed beamforming [shared] in : %lf\n", timestamp);

    // Run beamformer kernel
    cudaEventRecord(event_start, 0);
    CudaSafeCall(cudaMemcpy(output_buffer, d_output_buffer, nsamp * nchans * nbeams * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Copied results back to CPU memory in : %lf\n", timestamp);

    printf("Performance: %.2f Gflops\n", 32.0f * nchans * nsamp * nbeams * (nants * 0.25) * (1.0 / (kernel_time * 0.001)) * 1e-9);

//)* nbeams * nchans * nsamp * (((float)nants) / 4.0) * (1.0 / (kernel_time * 0.001)) * 1.0e-9);

    // Check to see if all output has been successful
    for(unsigned i = 0; i < nbeams; i++)
        for(unsigned j = 0; j < nchans; j++)
            for(unsigned k = 0; k < nsamp; k++)
                if ((output_buffer[i * nchans * nsamp + j * nsamp + k] - (j*j + j*i)*nants * 2) > 0.001)
                {
                    printf("!! %d.%d.%d = %f != %f\n", i, j, k, output_buffer[i * nchans * nsamp + j * nsamp + k],(j*j + j*j)*nants * 2);
                    exit(0);
                }
}

