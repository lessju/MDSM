#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <sys/time.h>
#include "time.h"
#include <math.h>

#define BEAMFORMER_THREADS 128
#define BEAMS 16
#define ANTS 32
unsigned nchans = 512, nants = 32, nsamp = 16384*2, nbeams = 16;

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

typedef struct
{
    char4 real1, real2, real3, real4, real5, real6, real7, real8;
    char4 imag1, imag2, imag3, imag4, imag5, imag6, imag7, imag8;
} ANTENNAS;

__global__ void beamform_struct(float4* input, float *output, float4 *shifts, unsigned nsamp,
                                unsigned nants, unsigned nchans, unsigned nbeams)
{
    // Loop over all time samples
    for(unsigned time = blockIdx.x;
                 time < nsamp;
                 time += gridDim.x)
    {
        // Synchronise threads
        __syncthreads();

        // Loop over channel
        for(unsigned channel = threadIdx.x;
                     channel < nchans;
                     channel += blockDim.x)
        {        
            // Synchronise threads
            __syncthreads();

            // Loop over all antennas (since we're compacting them into groups 0f 8,
            // we only need 2 iterations for 32 antennas)
            for(unsigned antenna = 0; antenna < nants / 8; antenna++)
            {
                register float4 real = input[time * nchans * 4 + channel * 4 + antenna];
                register float4 imag = input[time * nchans * 4 + channel * 4 + 2 + antenna];

                // Loop over all the beams
                float beam = 0;
                for(unsigned beam = 0; beam < nbeams; beam++)
                {
                    beam += real.w + imag.w;
                }

                // Save beam value to global memory
                output[blockIdx.x * blockDim.x + threadIdx.x] = beam;
            }
        }
    }	
}


typedef struct
{
    char a, b, c, d, e ,f, g, h, i, j, k, l, m, n, o, p;
} char16;

__global__ void beamform_struct_char(char16* input, float *output, float4 *shifts, unsigned nsamp,
                                unsigned nants, unsigned nchans, unsigned nbeams)
{
    // Loop over all time samples
    for(unsigned time = blockIdx.x;
                 time < nsamp;
                 time += gridDim.x)
    {
        // Synchronise threads
        __syncthreads();

        // Loop over channel
        for(unsigned channel = threadIdx.x;
                     channel < nchans;
                     channel += blockDim.x)
        {        
            // Synchronise threads
            __syncthreads();

            // Loop over all antennas (since we're compacting them into groups 0f 8,
            // we only need 2 iterations for 32 antennas)
            for(unsigned antenna = 0; antenna < nants / 8; antenna++)
            {
                register char16 real = input[time * nchans * 4 + channel * 4 + antenna];
                register char16 imag = input[time * nchans * 4 + channel * 4 + 2 + antenna];

                // Loop over all the beams
                float beam = 0;
                for(unsigned beam = 0; beam < nbeams; beam++)
                {
                    beam += real.a + imag.a;
                }

                // Save beam value to global memory
                output[blockIdx.x * blockDim.x + threadIdx.x] = beam;
            }
        }
    }	
}


// ==========================================================================
int main(int agrc, char *argv[])
{
    cudaSetDevice(0);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

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
    float *d_shifts;
    CudaSafeCall(cudaMallocHost((void **) &output_buffer, nchans * nsamp * nbeams * sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_output_buffer, nchans * nsamp * nbeams * sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_shifts, nchans * nbeams * nants * sizeof(float)));
    printf("Allocated output buffers\n");

    // Generate fake data
//    for(unsigned i = 0; i < nsamp; i++)
//        for(unsigned j = 0; j < nchans; j++)
//            for(unsigned k = 0; k < nants * 2; k++)
//                input_buffer[i * nchans * nants * 2 + j * nants * 2 + k] = j;
    memset(input_buffer, 1, nchans * nsamp * nants * sizeof(char2));
    printf("Generated fake data\n");

    // Generate shifts
    cudaMemset((void *) d_shifts, 1, nchans * nbeams * nants * sizeof(float));

    // Copy input buffer to GPU
    cudaEventRecord(event_start, 0);
    CudaSafeCall(cudaMemcpy(d_input_buffer, input_buffer, nchans * nsamp * nants * sizeof(char2), cudaMemcpyHostToDevice));
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Copied data to GPU in: %lf\n", timestamp);

    // Run beamformer kernel
    cudaEventRecord(event_start, 0);
    beamform_struct<<< 1024, BEAMFORMER_THREADS >>>
            ((float4 *) d_input_buffer, d_output_buffer,(float4 *) d_shifts, nsamp, nants, nchans, nbeams);
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
    double kernel_time = timestamp;
	printf("Performed beamforming [shared] in : %lf\n", timestamp);

    float flops = 29.0f * nchans * nsamp * nbeams * (nants * 0.25) * (1.0 / (kernel_time * 0.001)) * 1e-9;
    printf("Performance: %.2f Gflops (%.1f)\%\n", flops, flops / 2500.0 * 100);

    cudaEventRecord(event_start, 0);
    beamform_struct_char<<< 1024, BEAMFORMER_THREADS >>>
            ((char16 *) d_input_buffer, d_output_buffer,(float4 *) d_shifts, nsamp, nants, nchans, nbeams);
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
    kernel_time = timestamp;
	printf("Performed beamforming [shared] in : %lf\n", timestamp);


    // Run beamformer kernel
//    cudaEventRecord(event_start, 0);
//    CudaSafeCall(cudaMemcpy(output_buffer, d_output_buffer, nsamp * nchans * nbeams * sizeof(float), cudaMemcpyDeviceToHost));
//    cudaEventRecord(event_stop, 0);
//	cudaEventSynchronize(event_stop);
//	cudaEventElapsedTime(&timestamp, event_start, event_stop);
//	printf("Copied results back to CPU memory in : %lf\n", timestamp);

//    cudaEventRecord(event_start, 0);
//    CudaSafeCall(cudaMemcpy(input_buffer, d_input_buffer, nsamp * nchans * nants * sizeof(char2), cudaMemcpyDeviceToHost));
//    cudaEventRecord(event_stop, 0);
//	cudaEventSynchronize(event_stop);
//	cudaEventElapsedTime(&timestamp, event_start, event_stop);
//	printf("Copied results back to CPU memory in : %lf\n", timestamp);

    flops = 29.0f * nchans * nsamp * nbeams * (nants * 0.25) * (1.0 / (kernel_time * 0.001)) * 1e-9;
    printf("Performance: %.2f Gflops (%.1f)\%\n", flops, flops / 2500.0 * 100);


    // Check to see if all output has been successful
//    for(unsigned i = 0; i < nbeams; i++)
//        for(unsigned j = 0; j < nchans; j++)
//            for(unsigned k = 0; k < nsamp; k++)
//                if ((output_buffer[i * nchans * nsamp + j * nsamp + k] - (j*j + j*i)*nants * 2) > 0.001)
//                {
//                    printf("!! %d.%d.%d = %f != %f\n", i, j, k, output_buffer[i * nchans * nsamp + j * nsamp + k],(j*j + j*j)*nants * 2);
//                    exit(0);
//                }

    // Check to see if all output has been successful
//    for(unsigned i = 0; i < nsamp; i++)
//        for(unsigned j = 0; j < nchans; j++)
//            for(unsigned k = 0; k < nants * 2; k++)
//                if (input_buffer[i * nchans * nants * 2 + nchans * nants * 2 + k]!= 0)
//                {
//                    printf("!! %d.%d.%d = %d != %f\n", i, j, k, (int) input_buffer[i * nchans * nants * 2 + nchans * nants * 2 + k], 0.0f);
//                    exit(0);
//                }
}

