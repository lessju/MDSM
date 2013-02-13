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
unsigned nchans = 512, nants = 32, nsamp = 16384;

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

typedef struct
{
    char4 real[8];
    char4 imag[8];
} ANTENNAS;

__global__ void beamform_struct(ANTENNAS* input, float *output, float4 *shifts, unsigned nsamp,
                                unsigned nants, unsigned nchans)
{
    __shared__ float beams[BEAMFORMER_THREADS * BEAMS];

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
            // Initialise shared memory storing beams
            for(unsigned i = 0; i < BEAMS; i++)
                beams[blockDim.x * i + threadIdx.x] = 0;
        
            // Synchronise threads
            __syncthreads();

            // Load antennas
            ANTENNAS antennas = input[time * nchans + channel];

            // Loop over all antennas and beamform
            for(unsigned antenna = 0;
            			 antenna < nants / 4;
            			 antenna ++)
            {
                char4 real = antennas.real[antenna];
                char4 imag = antennas.imag[antenna];

                // Loop over all beams
				for(unsigned beam = 0; beam < BEAMS; beam++)
				{
                    // Add four antennas at a time (to reduce shared memory overhead and increase arithmetic intensity)
                    float4 phase_shifts = shifts[beam * nchans * nants / 4 + antenna * nchans + channel];

                    float temp1 = real.w * real.w + (phase_shifts.w * imag.w) * (phase_shifts.w * imag.w);
                    float temp2 = real.x * real.x + (phase_shifts.x * imag.x) * (phase_shifts.x * imag.x);
                    float temp3 = real.y * real.y + (phase_shifts.y * imag.y) * (phase_shifts.y * imag.y);
                    float temp4 = real.z * real.z + (phase_shifts.z * imag.z) * (phase_shifts.z * imag.z);
 					beams[blockDim.x * beam + threadIdx.x] += temp1 + temp2 + temp3 + temp4;
				}
            }

            // Synchronise threads
            __syncthreads();

            // Save beam value to global memory
            for(unsigned beam = 0; beam < BEAMS; beam++)
//                output[beam * nsamp * nchans + channel * nsamp + time] = beams[blockDim.x * beam + threadIdx.x];//beams[blockDim.x * beam + threadIdx.x];
                    output[blockIdx.x * blockDim.x * beam + threadIdx.x] = beams[beam];

            // Synchronise threads
            __syncthreads();
        }
    }	
}


__global__ void beamform_shared(char4 *input, float *output, float4 *shifts, unsigned nsamp,
                                unsigned nants, unsigned nchans)
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

        // Loop over channe
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
            for(unsigned i = 0; i < BEAMS; i++)
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
				for(unsigned beam = 0; beam < BEAMS; beam++)
				{
                    // Add four antennas at a time (to reduce shared memory overhead and increase arithmetic intensity)
                    float4 phase_shifts = shifts[beam * nchans * nants / 4 + antenna * nchans + channel];

                    float temp1 = real.w * real.w + (phase_shifts.w * imag.w) * (phase_shifts.w * imag.w);
                    float temp2 = real.x * real.x + (phase_shifts.x * imag.x) * (phase_shifts.x * imag.x);
                    float temp3 = real.y * real.y + (phase_shifts.y * imag.y) * (phase_shifts.y * imag.y);
                    float temp4 = real.z * real.z + (phase_shifts.z * imag.z) * (phase_shifts.z * imag.z);
 					beams[blockDim.x * beam + threadIdx.x] += temp1 + temp2 + temp3 + temp4;
				}
            }

            // Synchronise threads
            __syncthreads();

            // Save beam value to global memory
            for(unsigned beam = 0; beam < BEAMS; beam++)
//                output[beam * nsamp * nchans + channel * nsamp + time] = beams[blockDim.x * beam + threadIdx.x];//beams[blockDim.x * beam + threadIdx.x];
                    output[blockIdx.x * blockDim.x * beam + threadIdx.x] = beams[beam];

            // Synchronise threads
            __syncthreads();
        }
    }	
}
__global__ void 
__launch_bounds__(BEAMFORMER_THREADS) 
beamform_time(char4 *input, float *output, float *shifts, unsigned nsamp,
              unsigned nants, unsigned nchans)
{
    __shared__ char4   shared[BEAMFORMER_THREADS * 16];
    __shared__ float   phase_shifts[BEAMS * 4];

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
            shared[i] = input[index + (i / 16) * nchans * 16 + blockIdx.y * 16 + i % 16];

        // Synchronise threads
        __syncthreads();

        // Initialise beams registers
        register float beams[BEAMS] = {0};

        // Loop over all antennas
        for(unsigned antenna = 0; antenna < ANTS / 4; antenna++)
        {
            // Add four antennas at a time (to reduce shared memory overhead and increase arithmetic intensity)
            char4 real_char = shared[threadIdx.x * 16 + antenna];
            char4 imag_char = shared[threadIdx.x * 16 + antenna + 8];

            float realw = real_char.w;
            float imagw = imag_char.w;
            float realx = real_char.x; 
            float imagx = imag_char.x;
            float realy = real_char.y;
            float imagy = imag_char.y;
            float realz = real_char.z;
            float imagz = imag_char.z;

            // Load shifts associated with these four antennas and all beams
            for(unsigned i = threadIdx.x; i < 4 * BEAMS; i+= blockDim.x)
                phase_shifts[i] = shifts[(blockIdx.y * nants + antenna * 4) * BEAMS + i];

            // Synchronise threads
            __syncthreads();

            // Loop over all the beams
            for(unsigned beam = 0; beam < BEAMS; beam++)
            {
                // Read shifts from shared memory and apply to current four antennas
                float shift1 = phase_shifts[beam];
                float shift2 = phase_shifts[beam + BEAMS];
                float shift3 = phase_shifts[beam + BEAMS * 2];
                float shift4 = phase_shifts[beam + BEAMS * 3];

                float temp1 = realw * realw;
                float temp2 = (shift1 * imagw) * (shift1 * imagw);
                temp1 += realx * realx;
                temp2 += (shift2* imagx) * (shift2 * imagx);
                temp1 += realy * realy;
                temp2 += (shift3 * imagy) * (shift3 * imagy);
                temp1 += realz * realz;
                temp2 += (shift4 * imagz) * (shift4 * imagz);

                // Add value to beam in global memory
                beams[beam] += temp1 + temp2;
            }
        }

        for(unsigned beam = 0; beam < BEAMS; beam++)
            output[beam * nsamp * nchans + blockIdx.y * nsamp + time] = beams[beam];

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
//    cudaEventRecord(event_start, 0);
//    beamform_shared<<< 1024, BEAMFORMER_THREADS >>>
//            ((char4 *) d_input_buffer, d_output_buffer, (float4 *) d_shifts, nsamp, nants, nchans);
//    cudaEventRecord(event_stop, 0);
//	cudaEventSynchronize(event_stop);
//	cudaEventElapsedTime(&timestamp, event_start, event_stop);
//    double kernel_time = timestamp;
//	printf("Performed beamforming [shared] in : %lf\n", timestamp);

    cudaEventRecord(event_start, 0);
    beamform_time<<< dim3(64, nchans), BEAMFORMER_THREADS >>>
            ((char4 *) d_input_buffer, d_output_buffer, d_shifts, nsamp, nants, nchans);
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
    double kernel_time = timestamp;
	printf("Performed beamforming [time] in : %lf\n", timestamp);

//    cudaEventRecord(event_start, 0);
//    CudaSafeCall(cudaMemcpy(output_buffer, d_output_buffer, nsamp * nchans * BEAMS * sizeof(float), cudaMemcpyDeviceToHost));
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

    printf("Performance: %.2f Gflops\n", 24.0f * nchans * nsamp * BEAMS * (nants * 0.25) * (1.0 / (kernel_time * 0.001)) * 1e-9);

    // Check to see if all output has been successful
//    for(unsigned i = 0; i < BEAMS; i++)
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

