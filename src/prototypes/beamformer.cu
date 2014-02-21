#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <sys/time.h>
#include "time.h"
#include <math.h>

#define BEAMFORMER_THREADS 128
#define BEAMS_PER_TB 16
#define BEAMS 64
#define ANTS 32
unsigned nchans = 1024, nants = 32, nsamp = 8192;

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
__global__ void 
__launch_bounds__(BEAMFORMER_THREADS) 
beamformer(const char4 *input, float *output, const __restrict__ float2 *shifts, const unsigned nsamp, const unsigned nchans)
{   
    // Shared memory store for phase shifts
    // The inner-most loop will split antennas into groups of four, so we only need
    // BEAM_PER_TB * 8 floats per iteration
    __shared__ float2 coefficients[BEAMS_PER_TB * 4];

    // Threablock will loop over time for a single channel
    // Groups of beams change in the z-direction
    // Channel changes in the y-direction
    // Multiple blocks in the x-direction

    // Loop over time samples for current block
    for(unsigned time = blockIdx.x * blockDim.x + threadIdx.x;
                 time < nsamp;
                 time += gridDim.x * blockDim.x)
    {
        // Compute index to start of block
        unsigned index = (blockIdx.y * nsamp + blockIdx.x * blockDim.x) * ANTS / 4;

        // Initialise beam registers
        register float beams_real[BEAMS_PER_TB] = { 0 };
        register float beams_imag[BEAMS_PER_TB] = { 0 };

        // Loop over all antennas and compute phase components
        for(unsigned antenna = 0;
                     antenna < ANTS / 4;
                     antenna++)
        {
            // Load antenna values from global memory
            char4 antenna_val = input[index + threadIdx.x * ANTS / 4 + antenna];

            // Load shifts associated with these four antennas and all beams for current thread block
            for(unsigned i = threadIdx.x;
                         i < 4 * BEAMS_PER_TB;
                         i += blockDim.x)
                coefficients[i] = shifts[blockIdx.y * BEAMS * ANTS +
                                         antenna * 4 * BEAMS + blockIdx.z * BEAMS_PER_TB +
                                         i];

            float4 ant_real = {  (antenna_val.w >> 4) & 0xF,  (antenna_val.x >> 4) & 0xF,
                                 (antenna_val.y >> 4) & 0xF,  (antenna_val.z >> 4) & 0xF };
            float4 ant_imag = {  antenna_val.w & 0x0F,        antenna_val.x & 0x0F,
                                 antenna_val.y & 0x0F,        antenna_val.z & 0x0F };

            // Synchronise threads
            __syncthreads();

            // Loop over all beams
            for(unsigned beam = 0;
                         beam < BEAMS_PER_TB;
                         beam++)
            {
                float2 shift;

                shift = coefficients[beam];
                beams_real[beam] += ant_real.w * shift.x + ant_imag.w * shift.y;
                beams_imag[beam] += ant_imag.w * shift.x + ant_real.w * shift.y;

                shift = coefficients[BEAMS_PER_TB + beam];
                beams_real[beam] += ant_real.x * shift.x + ant_imag.x * shift.y;
                beams_imag[beam] += ant_imag.x * shift.x + ant_real.x * shift.y;

                shift = coefficients[2 * BEAMS_PER_TB + beam];
                beams_real[beam] += ant_real.y * shift.x + ant_imag.y * shift.y;
                beams_imag[beam] += ant_imag.y * shift.x + ant_real.y * shift.y;

                shift = coefficients[3 * BEAMS_PER_TB + beam];
                beams_real[beam] += ant_real.z * shift.x + ant_imag.z * shift.y;
                beams_imag[beam] += ant_imag.z * shift.x + ant_real.z * shift.y;
            }
        }

        // Add phase and amplitude parts and save computed beams to global memory
        for(unsigned beam = 0; beam < BEAMS_PER_TB; beam++)
            output[(blockIdx.z * BEAMS_PER_TB + beam) * nsamp * nchans + blockIdx.y * nsamp + time] = 
                 sqrt(beams_real[beam] * beams_real[beam] + beams_imag[beam] * beams_imag[beam]);

        // Synchronise threads
        __syncthreads();
    }
}


// ==========================================================================
int main(int agrc, char *argv[])
{
//    struct timeval start, end;
//    long mtime, seconds, useconds;

    cudaSetDevice(0);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    cudaFuncSetSharedMemConfig( beamformer, cudaSharedMemBankSizeEightByte );

	cudaEvent_t event_start, event_stop;
	float timestamp;
	cudaEventCreate(&event_start);
	cudaEventCreate(&event_stop);

    printf("Memory requirements: Input: %.2f MB, Output: %.2f MB \n", nchans * nsamp * nants * sizeof(char2) / (1024.0 * 1024),
                                                             nchans * nsamp * BEAMS * sizeof(float) / (1024.0 * 1024.0));
    printf("nsamp: %d, nchans: %d, BEAMS: %d, nants: %d\n", nsamp, nchans, BEAMS,  nants);

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
    float2 *d_shifts;
    CudaSafeCall(cudaMallocHost((void **) &output_buffer, nchans * nsamp * BEAMS * sizeof(float)));
    CudaSafeCall(cudaMallocHost((void **) &output_buffer, nchans * nsamp * BEAMS * sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_output_buffer, nchans * nsamp * BEAMS * sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_shifts, nchans * BEAMS * nants * sizeof(float2)));
    printf("Allocated output buffers\n");

    // Generate fake data
    for(unsigned i = 0; i < nsamp; i++)
        for(unsigned j = 0; j < nchans; j++)
            for(unsigned k = 0; k < nants * 2; k++)
                input_buffer[i * nchans * nants * 2 + j * nants * 2 + k] = j;
    memset(input_buffer, 1, nchans * nsamp * nants * sizeof(char2));
    printf("Generated fake data\n");

    // Generate shifts
    
    cudaMemset((void *) d_shifts, 1, nchans * BEAMS * nants * sizeof(float2));

    // Copy input buffer to GPU
    cudaEventRecord(event_start, 0);
    CudaSafeCall(cudaMemcpy(d_input_buffer, input_buffer, nchans * nsamp * nants * sizeof(unsigned char), cudaMemcpyHostToDevice));
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Copied data to GPU in: %lf\n", timestamp);

    // Run beamformer kernel
    cudaEventRecord(event_start, 0);
    beamformer<<< dim3(nsamp / BEAMFORMER_THREADS, nchans, BEAMS / BEAMS_PER_TB), BEAMFORMER_THREADS >>>
            ((char4 *) d_input_buffer, d_output_buffer, d_shifts, nsamp, nchans);
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
    double kernel_time = timestamp;
	printf("Performed beamforming [time] in : %lf\n", timestamp);

    float flops = 24.0f * nchans * nsamp * BEAMS * (nants * 0.25) * (1.0 / (kernel_time * 0.001)) * 1e-9;
    printf("Performance: %.2f Gflops (%.1f)\%\n", flops, flops / 2500.0 * 100);

    // Copy results back to GPU memory 
    cudaEventRecord(event_start, 0);
    CudaSafeCall(cudaMemcpy(output_buffer, d_output_buffer, nsamp * nchans * BEAMS * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Copied results back to CPU memory in : %lf\n", timestamp);

//    cudaEventRecord(event_start, 0);
//    CudaSafeCall(cudaMemcpy(input_buffer, d_input_buffer, nsamp * nchans * nants * sizeof(char2), cudaMemcpyDeviceToHost));
//    cudaEventRecord(event_stop, 0);
//	cudaEventSynchronize(event_stop);
//	cudaEventElapsedTime(&timestamp, event_start, event_stop);
//	printf("Copied results back to CPU memory in : %lf\n", timestamp);

    // Check to see if all output has been successful
    for(unsigned i = 0; i < BEAMS; i++)
        for(unsigned j = 0; j < nchans; j++)
            for(unsigned k = 0; k < nsamp; k++)
                if (abs(output_buffer[i * nchans * nsamp + j * nsamp + k] - 31.0)  > 0.001)
                {
                    printf("!! %d.%d.%d = %f != %f\n", i, j, k, output_buffer[i * nchans * nsamp + j * nsamp + k], 64.0);
                    exit(0);
                }

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

