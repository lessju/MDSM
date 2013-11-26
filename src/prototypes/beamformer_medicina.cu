#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include "time.h"
#include <math.h>

#define BEAMFORMER_THREADS 128
#define BEAMS_PER_TB 16
#define BEAMS 128
#define ANTS 32

unsigned nchans = 1024, nants = 32, nsamp = 32768/4;

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
        unsigned index = (blockIdx.y * nsamp + blockIdx.x) * ANTS / 4;
        
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

        // Loop over all antennas and compute phase components
        for(unsigned antenna = 0;
                     antenna < ANTS / 4;
                     antenna++)
        {
            // Add four antennas at a time (to reduce shared memory overhead and increase arithmetic intensity)
            char4 imag_char = shared[threadIdx.x * 8 + antenna];

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
				float shift; 

				shift        = phase_shifts[beam];
                beams[beam] += (shift * imagw) * (shift * imagw);
				shift        = phase_shifts[BEAMS_PER_TB + beam];
                beams[beam] += (shift * imagx) * (shift * imagx);
			    shift        = phase_shifts[2 * BEAMS_PER_TB + beam];
                beams[beam] += (shift * imagy) * (shift * imagy);
				shift        = phase_shifts[3 * BEAMS_PER_TB + beam];
                beams[beam] += (shift * imagz) * (shift * imagz);
            }
        }

        // Add phase and amplitude parts and save computed beams to global memory
        for(unsigned beam = 0; beam < BEAMS_PER_TB; beam++)
            output[(blockIdx.z * BEAMS_PER_TB + beam) * nsamp * nchans + blockIdx.y * nsamp + time] = 
                    beams[beam] + real[threadIdx.x];

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

	cudaEvent_t event_start, event_stop;
	float timestamp;
	cudaEventCreate(&event_start);
	cudaEventCreate(&event_stop);

    printf("Memory requirements: Input: %.2f MB, Output: %.2f MB \n", nchans * nsamp * nants * sizeof(char) / (1024.0 * 1024),
                                                                      nchans * nsamp * BEAMS * sizeof(float) / (1024.0 * 1024.0));
    printf("nsamp: %d, nchans: %d, BEAMS: %d, nants: %d\n", nsamp, nchans, BEAMS,  nants);

    // Allocate and initialise CPU and GPU memory for data
    // Data is stored in frequency/time/antenna order with antenna changing the fastest
    // Each complex value is packed into one byte [RRRRIIII]
    char *input_buffer, *d_input_buffer;
  CudaSafeCall(cudaHostAlloc((void **) &input_buffer, nchans * nsamp * nants * sizeof(char), cudaHostAllocMapped));
//    input_buffer = (char *) malloc(nchans * nsamp * nants * sizeof(char));

    // We will be outputing beam to be processed by the transient detection pipeline,
    // whose required input data format is beam/channel/time, with time changing the faster,
    // and is in 32-bit single precision floating point
    float *d_output_buffer, *output_buffer;
    float *d_shifts, *shifts;
//    output_buffer = (float *) malloc(nchans * nsamp * BEAMS * sizeof(float));
//    shifts = (float *) malloc(nchans * BEAMS * nants * sizeof(float));
    CudaSafeCall(cudaHostAlloc((void **) &output_buffer, nchans * nsamp * BEAMS * sizeof(float), cudaHostAllocMapped));
	CudaSafeCall(cudaHostAlloc((void **) &shifts, BEAMS * nants * nchans * sizeof(float), 0));


    // Generate fake data
//	for(unsigned i = 0; i < nchans; i++)
//		for(unsigned j = 0; j < nsamp; j++)
//			for(unsigned k = 0; k < nants; k++)
//				input_buffer[i * nsamp * nants + j * nants + k] = 0x11;

//    printf("Generated fake data\n");

    // Generate shifts
    for(unsigned i = 0; i < nchans; i++)
        for(unsigned j = 0; j < nants * BEAMS; j++)
    		shifts[i * nants * BEAMS + j] = i;

    // Allocate GPU buffers
    CudaSafeCall(cudaMalloc((void **) &d_input_buffer, nchans * nsamp * nants * sizeof(char)));
    CudaSafeCall(cudaMalloc((void **) &d_output_buffer, nchans * nsamp * BEAMS * sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_shifts, nchans * BEAMS * nants * sizeof(float)));
    CudaSafeCall(cudaMemset(d_output_buffer, 0, nchans * nsamp * BEAMS * sizeof(float)));

    // Copy input buffer to GPU
    cudaEventRecord(event_start, 0);
    CudaSafeCall(cudaMemcpy(d_input_buffer, input_buffer, nchans * nsamp * nants * sizeof(char), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_shifts, shifts, nchans * nants * BEAMS * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied data to GPU in: %lf\n", timestamp);

	// Run beamformer kernel
	cudaEventRecord(event_start, 0);
    beamform_medicina<<< dim3(nsamp / BEAMFORMER_THREADS, nchans, BEAMS / BEAMS_PER_TB), BEAMFORMER_THREADS >>>
					     ((char4 *) d_input_buffer, d_output_buffer, d_shifts, nsamp, nchans);
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
    double kernel_time = timestamp;
	double flops = 17.0f * nchans * nsamp * BEAMS * (nants * 0.25) * (1.0 / (kernel_time * 0.001)) * 1e-9;
	printf("Performed beamforming - Medicina [%.2f Gflops (%.1f)] in : %lf\n", flops, flops / 3500.0 * 100, timestamp);

    // Copy output to CPU memory
    cudaEventRecord(event_start, 0);
    CudaSafeCall(cudaMemcpy(output_buffer, d_output_buffer, nsamp * nchans * BEAMS * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied results back to CPU memory in : %lf\n", timestamp);


    // Check to see if all output has been successful
//    for(unsigned i = 0; i < BEAMS; i++)
//        for(unsigned j = 0; j < nchans; j++)
//            for(unsigned k = 0; k < nsamp; k++)
//				 if ((output_buffer[i * nchans * nsamp + j * nsamp + k] - (32.0 + j * j * 32.0)) > 0.001)
//				{
//                    printf("!! %d.%d.%d = %f != %f\n", i, j, k, output_buffer[i * nchans * nsamp + j * nsamp + k], 32.0 + j * j * 32.0);
//                    exit(0);
//                }
}
