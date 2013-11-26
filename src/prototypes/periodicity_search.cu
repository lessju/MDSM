#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <sys/time.h>
#include "time.h"
#include <math.h>
#include "unistd.h"
#include "file_handler.h"
#include "cufft.h"
 

#define BINNING_THREADS 256

//char *filename = "/data/Data/SETI/B1839+56_8bit.fil";
//char *filename = "/data/Data/SETI/samplePulsar.fil";
char *filename = "/home/lessju/Kepler_Pulsar_RFI.dat";
int nsamp = 32768, ndms = 4096;

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

// ========================= CPU HELPER FUNCTIONS ===========================
void read_data(float *buffer, unsigned nsamp, unsigned nchans)
{
    // Read file
    float *tempBuff = (float *) malloc(nsamp * nchans * sizeof(float));
    FILE *fp = fopen(filename, "rb");
    
    // Read header
    read_header(fp);

    int num_read = read_block(fp, 32, tempBuff, nchans * nsamp);
    fclose(fp);

    // Transpose data
    unsigned i, j;
    for(i = 0; i < nchans; i++)    
        for(j = 0; j < nsamp; j++)
            buffer[i * nsamp + j] = tempBuff[j * nchans + i];

    free(tempBuff);

    if (num_read != nsamp * nchans)
    {
        printf("Seems there's not enough data in the file\n");
        exit(0);
    }
}

// Process command-line parameters
void process_arguments(int argc, char *argv[])
{
    int i = 1;
    
    while((fopen(argv[i], "r")) != NULL)
        i++;

    while(i < argc) {
       if (!strcmp(argv[i], "-ndms"))
           ndms = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-nsamp"))
           nsamp = atoi(argv[++i]);
       i++;
    }
}

// ==========================================================================
// Estimate power of half-integer frequencies from 2 neighboring bins
__global__ void interbinning(cufftComplex *input, unsigned nsamp)
{
   __shared__ cufftComplex local_store[BINNING_THREADS + 2];

    // First iteration, set first local store value to first nsamp value
    if (threadIdx.x == 0)
        local_store[0] = input[blockDim.y * nsamp];

    __syncthreads();

    // Each block will process the entire input array for a single DM value
    for(unsigned i = threadIdx.x;
                 i < nsamp; 
                 i += blockDim.x)
    {
        // Load current buffer
        local_store[threadIdx.x + 1] = input[blockDim.y * nsamp + i];

        // Load extra value at the end of the buffer
        if (threadIdx.x == blockDim.x - 1)
            local_store[BINNING_THREADS + 1] = input[blockDim.y * nsamp + BINNING_THREADS + 1];

        // Calculate proper value and save to buffer
        cufftComplex A = local_store[threadIdx.x], 
                     B = local_store[threadIdx.x + 1], 
                     C = local_store[threadIdx.x + 2],
                    temp1, temp2;

        temp1.x = A.x + B.x;
        temp1.y = A.y + B.y;
        temp2.x = B.x + C.x;
        temp2.y = B.y + C.y;
            
        input[blockDim.y * nsamp + i].x = max(temp1.x * temp1.x + temp1.y * temp1.y, 
                                          max(B.x * B.x + B.y * B.y, 
                                              temp2.x * temp2.x + temp2.y * temp2.y));
    }

}


// ==========================================================================

// Main function
int main(int argc, char *argv[])
{
    struct timeval start, end;
    long mtime, seconds, useconds; 

    cudaSetDevice(1);

	cudaEvent_t event_start, event_stop;
	float timestamp;
	cudaEventCreate(&event_start); 
	cudaEventCreate(&event_stop); 

    // Allocate and initialise CPU and GPU memory for data
    float *input_buffer, *d_input;
    cufftComplex *d_output, *output_buffer;
    CudaSafeCall(cudaMallocHost((void **) &input_buffer, nsamp * ndms * sizeof(float), cudaHostAllocPortable));
    CudaSafeCall(cudaMallocHost((void **) &output_buffer, (nsamp / 2 + 1) * ndms * sizeof(cufftComplex), cudaHostAllocPortable));
    CudaSafeCall(cudaMalloc((void **) &d_input, nsamp * ndms * sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_output, ((nsamp / 2) + 1) * ndms * sizeof(cufftComplex)));
    CudaSafeCall(cudaMemset(d_output, 0, ((nsamp / 2) + 1) * ndms * sizeof(cufftComplex)));

    printf("nsamp: %d, ndms: %d, input: %.2f MB, output: %.2f MB\n", nsamp, ndms, (nsamp * ndms / (1024*1024.0)) * sizeof(float), 
                                                                              ((nsamp / 2 + 1) * ndms / (1024*1024.0)) * sizeof(float));

    // Initialise input data (sine wave with DM frequency) (temporary)
    for(unsigned i = 0; i < ndms; i++)
        for(unsigned j = 0; j < nsamp; j++)
            input_buffer[i * nsamp + j] = sin(2 * M_PI * (i*32) / nsamp * j);

    // Copy input buffer to GPU memory
    cudaEventRecord(event_start, 0);
    cudaMemcpy(d_input, input_buffer, nsamp * ndms * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Copied data to GPU in: %lf\n", timestamp);

    // Create FFT plan and perform FFTs
    cudaEventRecord(event_start, 0);
    cufftHandle plan;
    cufftResult result;
    if ((result = cufftPlan1d(&plan, nsamp, CUFFT_R2C, ndms)) != CUFFT_SUCCESS)
    {
        switch (result)
        {
            case CUFFT_SETUP_FAILED:
                printf("CUFFT error: CUFFT library failed to initialise!\n");                
            case CUFFT_INVALID_SIZE:
                printf("CUFFT error: The nx parameter is not a supported size!\n");
            case CUFFT_INVALID_TYPE:
                printf("CUFFT error: The type parameter is not supported!\n"); 
            case CUFFT_ALLOC_FAILED:    
                printf("CUFFT error: Allocation of GPU resources for the plan failed\n");
        }
        exit(-1);

    }

    if (cufftExecR2C(plan, (cufftReal *) d_input, (cufftComplex *)d_input) != CUFFT_SUCCESS)
        printf("CUFFT error: ExecC2C failed!\n");

    cudaThreadSynchronize();

    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Performed FFTs: %lf\n", timestamp);

    sleep(10);
    cufftDestroy(plan);

    // Copy result back to CPU memory
    cudaEventRecord(event_start, 0);
    CudaSafeCall(cudaMemcpy(output_buffer, d_output, ((nsamp / 2) + 1) * ndms * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Copied results back to CPU memory in : %lf\n", timestamp);

    sleep(10);

    // Perform inter-binning
//    cudaEventRecord(event_start, 0);
//    interbinning<<<dim3(1, ndms), BINNING_THREADS>>>((cufftComplex *) d_input, nsamp);
//    cudaEventRecord(event_stop, 0);
//	cudaEventSynchronize(event_stop);
//	cudaEventElapsedTime(&timestamp, event_start, event_stop);
//	printf("Performed interbinning in: %lf\n", timestamp);

    // Compute component power and dump to disk
    unsigned ncomps = nsamp / 2.0 + 1;
    float *output = (float *) malloc(ncomps * ndms * sizeof(float));
    for(unsigned i = 0; i < ndms; i++)   
        for(unsigned j = 0; j < ncomps; j++)
            output[i * ncomps + j] = output_buffer[i * ncomps + j].x * output_buffer[i * ncomps + j].x + output_buffer[i * ncomps + j].y * output_buffer[i * ncomps + j].y;

    // Write buffer to file
    FILE *fp = fopen("Test_periodicity.dat", "wb");
    fwrite(output, ndms * ncomps, sizeof(float), fp);
    fclose(fp);
}

    
