
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "time.h"
#include "string.h"
#include "math.h"

#include "cufft.h"

#define NTAPS 32

unsigned nsubs = 32, nsamp = 65536, nbeams = 1;
unsigned nchans = 1024;

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

inline void Exit()
{
	exit(0);
}

// ======================= Channelisation Kernels =============================
// NOTE: For this kernel, nchans <= blockDim.x
__global__ void ppf_fir(cuComplex *input, const float *window, const unsigned nsamp, 
                        const unsigned nsubs, const unsigned nbeams, const unsigned nchans) 
{
    // Subband moves in y dimension
    // Beam moves in z dimension

    // Declare shared memory to store window coefficients
     extern __shared__ float coeffs[];

    // Each thread is associated with a particular channel and sample
    unsigned channel_num = threadIdx.x % nchans;
    unsigned sample_num = threadIdx.x / nchans;
    unsigned sample_shift = (blockDim.x / nchans) == 0 ? 1 : blockDim.x / nchans;

    // Loop over channels (in cases where nchans > blockDim.x)
    for(unsigned c = channel_num;
                 c < nchans;
                 c += blockDim.x)
    {
        // FIFO buffer is stored in local register array
        cuComplex fifo[NTAPS] = { 0 };  

        // Initialise FIFO with first NTAPS values
        unsigned index = blockIdx.y * nsubs * nsamp * nchans + blockIdx.x * nsamp * nchans;
        for(unsigned i = 0; i < NTAPS - 1; i++)
            fifo[i] = input[index + nchans * i + c];

        // Load window coefficients to be used by each thread
        for(unsigned i = 0; i < NTAPS; i++)
            coeffs[threadIdx.x + i * blockDim.x] = window[i * nchans + c];

        // Synchronise threads
        __syncthreads();

        // Loop over all samples for current channel
        // Start at the (NTAPS-1)th sample, in order to use FIFO buffer
		#pragma unroll 2
        for(unsigned s = sample_num + NTAPS - 1;
                     s < nsamp;
                     s += sample_shift)
        {
            // Declare output value
            cuComplex output = { 0, 0 };

            // Store new value in FIFO buffer
            fifo[NTAPS - 1] = input[index + s * nchans + c];

            // Apply window
			#pragma unroll NTAPS
            for (unsigned t = 0; t < NTAPS; t++)
            {
                float coeff = coeffs[threadIdx.x + blockDim.x * t];
                output.x += fifo[t].x * coeff;
                output.y += fifo[t].y * coeff;
            }

			// Store output to global memory
            input[index + s * nchans + c] = output;

            // Re-arrange FIFO buffer
			#pragma unroll NTAPS
			for(unsigned i = 0; i < NTAPS - 1; i++)
				fifo[i] = fifo[i + 1];
        } 
    }
}

__global__ void fix_channelisation(float2 *input, float *output, unsigned nsamp, unsigned nchans, unsigned nbeams, 
                                   unsigned subchans, unsigned start_chan)
{    
    // Time changes in the x direction
    // Channels change along the y direction. Indexing start at start_chan
    // Beams change along the z direction
    // Each thread processes one sample

	// Get index to start of current channelised block
    // ThreadIdx.x is the nth channel formed in this block
	unsigned long indexIn  = blockIdx.z * nchans * nsamp + (start_chan + blockIdx.y) * nsamp + threadIdx.x;
    unsigned long indexOut =  (blockIdx.y * subchans + threadIdx.x) * nbeams + blockIdx.z;

    for(unsigned s = blockIdx.x;
                 s < nsamp / subchans;
                 s += gridDim.x)
    {
        float2 value = input[indexIn + s * subchans];
        output[s * nbeams * gridDim.y * subchans + indexOut] = sqrtf(value.x * value.x + value.y * value.y);
    }
}

// ======================= Main Program =======================================

// Process command-line parameters
void process_arguments(int argc, char *argv[])
{
    int i = 1;
    
    while(i < argc) {
       if (!strcmp(argv[i], "-nchans"))
           nchans = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-nsamp"))
           nsamp = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-nbeams"))
           nbeams = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-nsubs"))
           nsubs = atoi(argv[++i]);
       i++;
    }
}

// Notes: in real-time application we'll need to buffer the last ntaps * nchans values
// for use in the next buffer. Strategy:
// - Keep extra values in a separate buffer (multiple device memcopies)
// - Use this buffer to load the initial values, normal data is loaded in main loop
int main(int argc, char *argv[])
{
    cuComplex *input, *d_input;
	float *output, *d_output, *weights, *d_weights;

    process_arguments(argc, argv);

    printf("nsamp: %d, nsubs: %d, nbeams: %d, nchans: %d\n", nsamp, nsubs, nbeams, nchans);
    printf("Memory requirements: Input: %.2f MB, Output: %.2f \n", nsubs * nbeams * nsamp * sizeof(cuComplex) / (1024.0 * 1024), 0);

	// Set 8-byte shared memory
//	cudaFuncSetSharedMemConfig(ppf_fir, cudaSharedMemBankSizeEightByte );

    // Allocate and initialise arrays
    input   = (cuComplex *) malloc(nsamp * nsubs * nbeams * sizeof(cuComplex));
	output  = (float *) malloc(nsamp * nsubs * nbeams * sizeof(float));
    weights = (float *) malloc(nchans * NTAPS * sizeof(float));
    memset(input,  0, nsamp * nsubs * nbeams * sizeof(cuComplex));
	memset(output, 0, nsamp * nsubs * nbeams * sizeof(float));
    memset(weights, 0, nchans * NTAPS * sizeof(float));

	// Load coefficients
    char filename[256];
    sprintf(filename, "coeff_%d_%d.dat", NTAPS, nchans);
    FILE *fp = fopen(filename, "rb");
    fread(weights, sizeof(float), NTAPS * nchans, fp);

	// Initialise inputs (generate sin wave in each subband
    srand(time(NULL));
    for(unsigned b = 0; b < nbeams; b++)
        for(unsigned sb = 0; sb < nsubs; sb++)
            for(unsigned i = 0; i < nsamp; i++)
            {
                input[b * nsubs * nsamp + sb * nsamp + i].x = sin(i * 0.1) + sin(i*0.5);// + rand() * 1e-9;
                input[b * nsubs * nsamp + sb * nsamp + i].y = 0;
            }

    // Write input file
    FILE *inFile = fopen("input.dat", "wb");
    fwrite(input, sizeof(cuComplex), nbeams * nsubs * nsamp, inFile);
    fclose(inFile);

    // Initialise CUDA stuff
    CudaSafeCall(cudaSetDevice(1));
	CudaSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    cudaEvent_t event_start, event_stop;
    float timestamp;

    cudaEventCreate(&event_start); 
    cudaEventCreate(&event_stop);

    // Allocate GPU memory
    CudaSafeCall(cudaMalloc((void **) &d_input, nsamp * nsubs * nbeams * sizeof(cuComplex)));
	CudaSafeCall(cudaMalloc((void **) &d_output, nsamp * nsubs * nbeams * sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_weights, NTAPS * nchans * sizeof(float)));

    time_t start = time(NULL);

    // Copy input to GPU
    cudaEventRecord(event_start, 0);
    CudaSafeCall(cudaMemcpy(d_input, input, nsamp * nbeams * nsubs * sizeof(cuComplex), cudaMemcpyHostToDevice));    
	CudaSafeCall(cudaMemcpy(d_weights, weights, nchans * NTAPS * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied to GPU in: %lf \n", timestamp); 

    // Phase 1, perform FIR (apply window)
    cudaEventRecord(event_start, 0);
    unsigned num_threads = 128;
    dim3 grid(nsubs, nbeams);
    ppf_fir<<<grid, num_threads, NTAPS * num_threads * sizeof(float)>>>(d_input, d_weights, nsamp / nchans, nsubs, nbeams, nchans);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Performed FIR in: %lf [%.2f Gflops] \n", timestamp, (4 * NTAPS * nsubs * (nbeams * nsamp * 1.0e-9)) * (1.0 / (timestamp * 0.001)));

	// Phase 2, perform FFT
	cufftHandle plan;
    cufftPlan1d(&plan, nchans, CUFFT_C2C, nsubs * nsamp / nsubs); // Plan only created once
   
	cudaEventRecord(event_start, 0);
	for (unsigned i = 0; i < nbeams; i++)
        cufftExecC2C(plan, d_input + i * nsubs * nsamp, d_input + i * nsubs * nsamp, CUFFT_FORWARD);
	cudaThreadSynchronize();

	cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Performed FFT in: %lf\n", timestamp);

    cufftDestroy(plan);

	// Phase 3, fix channelisation order
//	cudaEventRecord(event_start, 0);
//	dim3 fixDim(nsamp / nchans, nsubs, nbeams);  
//    fix_channelisation<<< fixDim, nchans >>> 
//                      (d_input, d_output, nsamp, nsubs, nbeams, nsubs, 0);
//	cudaEventRecord(event_stop, 0);
//    cudaEventSynchronize(event_stop);
//    cudaEventElapsedTime(&timestamp, event_start, event_stop);
//	printf("Reordered channels in: %lf [%.2f Gflops]\n", timestamp,  (10 * nsubs * (nbeams * nsamp * 1.0e-9)) * (1.0 / (timestamp * 0.001)));

    // Copy output from GPU
    cudaEventRecord(event_start, 0);
    CudaSafeCall(cudaMemcpy(input, d_input, nsamp * nbeams * nsubs * sizeof(cuComplex), cudaMemcpyDeviceToHost));    
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied to GPU in: %lf \n", timestamp);

    // Write test output file
    FILE *outFile = fopen("output.dat", "wb");
    fwrite(input, sizeof(cuComplex), nbeams * nsubs * nsamp, outFile);
    fclose(outFile);

    Exit();
}

