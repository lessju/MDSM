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
#define BEAMS 32
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

// Kernel which paralellises over time instead of frequency within the blocks
__global__ void 
__launch_bounds__(BEAMFORMER_THREADS) 
beamform_time(char4 *input, float *output, float *shifts, unsigned nsamp,
              unsigned nants, unsigned nchans)
{
    __shared__ char4   shared[BEAMFORMER_THREADS * 8];
    __shared__ float   real[BEAMFORMER_THREADS];
    __shared__ float   phase_shifts[BEAMS_PER_TB * 4];

    // Thread block will loop over time for a single channel
    // Channel changes in the y direction
    // Multiple blocks in the x direction

    // Loop over time samples for current block
    for(unsigned time = blockIdx.x * blockDim.x + threadIdx.x;
                 time < nsamp;
                 time += gridDim.x * blockDim.x)
    {
        // Load data to shared memory
        unsigned index = (time / blockDim.x) * blockDim.x * nchans * 16;

        #pragma unroll 8
        for(unsigned i = threadIdx.x; i < blockDim.x * 16; i += blockDim.x)
        {
            // Grab some antenna data from global memory
            char4 value = input[index + (i / 16) * nchans * 16 + blockIdx.y * 16 + i % 16];

            // The first eight threads in a half-warp will contains the real components
            if (i % 16 < 8)
            {
                // First, combine current antennas
                float4 real_value = { value.w, value.x, value.y, value.z };
                float  curr_value = real_value.w * real_value.w + real_value.x * real_value.x + real_value.y * real_value.y + real_value.z * real_value.z;

                // We need to combine antennas from 8 different threads... use warp shuffle!
                curr_value += __shfl_down(curr_value, 1, 2);
                curr_value += __shfl_down(curr_value, 2, 4);
                curr_value += __shfl_down(curr_value, 4, 8);

                // Write final value to shared memory
                if (i % 16 == 0)
                    real[i / 16] = curr_value;
            }
            // The second eight threads in a half-warp will contain the imaginary components
            else
                shared[(i / 8) - 1 + (i - 8) % 8] = value;
        }

        // Synchronise threads
        __syncthreads();

        // Initialise beams registers
        register float beams[BEAMS_PER_TB] = {0};

        // Loop over all antennas
        for(unsigned antenna = 0; antenna < ANTS / 4; antenna++)
        {
            // Add four antennas at a time (to reduce shared memory overhead and increase arithmetic intensity)
            char4 imag_char  = shared[threadIdx.x * 8 + antenna];
            float real_value = real[threadIdx.x];

            float imagw = imag_char.w;
            float imagx = imag_char.x;
            float imagy = imag_char.y;
            float imagz = imag_char.z;

            // Load shifts associated with these four antennas and all beams
            for(unsigned i = threadIdx.x; i < 4 * BEAMS_PER_TB; i+= blockDim.x)
                phase_shifts[i] = shifts[blockIdx.y * BEAMS * nants + antenna * 4 * BEAMS + blockIdx.z * BEAMS_PER_TB + i];

            // Synchronise threads
            __syncthreads();

            // Loop over all the beams
            for(unsigned beam = 0; beam < BEAMS_PER_TB; beam++)
            {
                // Read shifts from shared memory and apply to current four antennas
                float shift1 = phase_shifts[beam];
                float shift2 = phase_shifts[BEAMS_PER_TB + beam];
                float shift3 = phase_shifts[2 * BEAMS_PER_TB + beam];
                float shift4 = phase_shifts[3 * BEAMS_PER_TB + beam];

                float temp2 = (shift1 * imagw) * (shift1 * imagw);
                temp2 += (shift2* imagx) * (shift2 * imagx);
                temp2 += (shift3 * imagy) * (shift3 * imagy);
                temp2 += (shift4 * imagz) * (shift4 * imagz);

                // Add value to beam in global memory
                beams[beam] += real_value + temp2;
            }
        }

        for(unsigned beam = 0; beam < BEAMS_PER_TB; beam++)
            output[(blockIdx.z * BEAMS_PER_TB + beam) * nsamp * nchans + blockIdx.y * nsamp + time] = beams[beam];

        // Synchronise threads
        __syncthreads();
    }
}

// Kernel which paralellises over time instead of frequency within the blocks
// Medicina implementation... this assumes 32 antennas
__global__ void 
__launch_bounds__(BEAMFORMER_THREADS) 
beamform_medicina(char4 *input, float *output, float *shifts, unsigned nsamp, unsigned nchans)
{
    // Shared memory store for imaginary components (BEAMFORMER_THREADS * 8 [*4 for char4 implied])   
    __shared__ char4 shared[BEAMFORMER_THREADS * 8];

    // Shared memory store for combined real components (one combined value per thread)
    __shared__ float real[BEAMFORMER_THREADS];

    // Shared memory store for phase shifts
    // The inner-most loop will split antennas into groups of four, so we only need
    // BEAM_PER_TB * 4 float per iteration
    __shared__ float phase_shifts[BEAMS_PER_TB * 4];

    // Threablock will loop over time for a single channel
    // Groups of beams change in the z-direction
    // Channel changes in the y-direction
    // Multiple blocks in the x-direction
    
    // Loop over time samples for current block
    for(unsigned time = blockIdx.x * blockDim.x + threadIdx.x;
                 time < nsamp;
                 time += gridDim.x * blockDim.x)
    {
        // Load antenna data for current spectra subset (BEAMFORMER_THREADS in all) to shared
        // memory, placing phase components in shared and combined amplitudes in real

        // Compute index to start of block
        unsigned index = (blockIdx.y * nsamp + blockIdx.x) * ANTS;
        
        // Loop over antennas in groups of four
        for(unsigned i = threadIdx.x;
                     i < blockDim.x * 8; // One memory request will load 4 full antennas
                     i += blockDim.x)
        {
            // Grab 4 antennas from global memory. This a quarter warp will contain all the antenna values
            // TODO: Check if this is correct
            register char4 value = input[index + i];

            // Each warp make up a single spectrum. Combine the amplitude components and store in real
            float4 real_value = { (value.w >> 4) & 0xF, (value.x >> 4) & 0xF, 
                                  (value.y >> 4) & 0xF, (value.z >> 4) & 0xF };

            // Combine antennas belonging to the current thread
            float   amplitude = real_value.w * real_value.w + real_value.x * real_value.x +
                                real_value.y * real_value.y + real_value.z * real_value.z;

            
            // Use warp-shuffle to combine antennas from 8 different threads
            amplitude += __shfl_down(amplitude, 1, 2);
            amplitude += __shfl_down(amplitude, 2, 4);
            amplitude += __shfl_down(amplitude, 4, 8);

            // Each 8th thread will contain a valid ampltiude value. Store this to real
            if (i % 8 == 0)
                real[i / 8] = amplitude;

            // We are ready from processing the amplitude value. Next we just need to store the
            // phase components to shared
            value.w = value.w & 0x0F;
            value.x = value.x & 0x0F;
            value.y = value.y & 0x0F;
            value.z = value.z & 0x0F;

            shared[i] = value;
        }        

        // Finished pre-computation. Synchronise threads
        __syncthreads();

        // Initialise beam registers
        register float beams[BEAMS_PER_TB] = { 0 };

        // Loop over all antennas
        for(unsigned antenna = 0;
                     antenna < ANTS / 4;
                     antenna++)
        {
            // Add four antennas at a time (to reduce shared memory overhead and increase arithmetic intensity)
            char4 imag_char = shared[threadIdx.x * 8 + antenna];
            float real_val  = real[threadIdx.x];

            float imagw = imag_char.w;
            float imagx = imag_char.x;
            float imagy = imag_char.y;
            float imagz = imag_char.z;

            // Load shifts associated with these four antennas and all beams for current thread block
            for(unsigned i = threadIdx.x; 
                         i < 4 * BEAMS_PER_TB; 
                         i += blockDim.x)
                phase_shifts[i] = shifts[blockIdx.y * BEAMS * ANTS + 
                                         antenna * 4 * BEAMS + blockIdx.z * BEAMS_PER_TB + 
                                         i];

            // Synchronise threads
            __syncthreads();
            
            // Loop over all beams 
            for(unsigned beam = 0;
                         beam < BEAMS_PER_TB;
                         beam++)
            {
                // Read shifts from shared memory and apply to current four antennas
                float shift1 = phase_shifts[beam];
                float shift2 = phase_shifts[BEAMS_PER_TB + beam];
                float shift3 = phase_shifts[2 * BEAMS_PER_TB + beam];
                float shift4 = phase_shifts[3 * BEAMS_PER_TB + beam];

                float temp2 = (shift1 * imagw) * (shift1 * imagw);
                temp2 += (shift2 * imagx) * (shift2 * imagx);
                temp2 += (shift3 * imagy) * (shift3 * imagy);
                temp2 += (shift4 * imagz) * (shift4 * imagz);

                // Add value to beam in beam registers
                // TODO: Check if this is correct
               beams[beam] += real_val * real_val + temp2 * temp2;
            }
        }

        // Save computed beams to global memory
        for(unsigned beam = 0; beam < BEAMS_PER_TB; beam++)
            output[(blockIdx.z * BEAMS_PER_TB + beam) * nsamp * nchans + blockIdx.y * nsamp + time] = beams[beam];

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
    cudaFuncSetSharedMemConfig( beamform_time, cudaSharedMemBankSizeEightByte );

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
    float *d_shifts;
    CudaSafeCall(cudaMallocHost((void **) &output_buffer, nchans * nsamp * BEAMS * sizeof(float)));
    CudaSafeCall(cudaMallocHost((void **) &output_buffer, nchans * nsamp * BEAMS * sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_output_buffer, nchans * nsamp * BEAMS * sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_shifts, nchans * BEAMS * nants * sizeof(float)));
    printf("Allocated output buffers\n");

    // Generate fake data
//    for(unsigned i = 0; i < nsamp; i++)
//        for(unsigned j = 0; j < nchans; j++)
//            for(unsigned k = 0; k < nants * 2; k++)
//                input_buffer[i * nchans * nants * 2 + j * nants * 2 + k] = j;
    memset(input_buffer, 1, nchans * nsamp * nants * sizeof(char2));
    printf("Generated fake data\n");

    // Generate shifts
    
    cudaMemset((void *) d_shifts, 1, nchans * BEAMS * nants * sizeof(float));

    // Copy input buffer to GPU
    cudaEventRecord(event_start, 0);
    CudaSafeCall(cudaMemcpy(d_input_buffer, input_buffer, nchans * nsamp * nants * sizeof(char2), cudaMemcpyHostToDevice));
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Copied data to GPU in: %lf\n", timestamp);

    // Run beamformer kernel
    cudaEventRecord(event_start, 0);
    beamform_time<<< dim3(nsamp / BEAMFORMER_THREADS, nchans, BEAMS / BEAMS_PER_TB), BEAMFORMER_THREADS >>>
            ((char4 *) d_input_buffer, d_output_buffer, d_shifts, nsamp, nants, nchans);
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
                if (abs(output_buffer[i * nchans * nsamp + j * nsamp + k] - 32.0)  > 0.001)
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

