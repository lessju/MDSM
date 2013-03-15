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
 

#define BANDPASS_THREADS 64

//char *filename = "/data/Data/SETI/B1839+56_8bit.fil";
//char *filename = "/data/Data/SETI/samplePulsar.fil";
char *filename = "/home/lessju/Kepler_Pulsar_RFI.dat";
//char *filename = "/home/lessju/Medicina_Channel_RFI_and_Pulsar.dat";
//char *filename = "/home/lessju/Medicina_Time_RFI.dat";
//char *filename = "/home/lessju/Medicina_Channel_RFI_and_Pulse.dat";
//char *filename  = "/tmp/0.dat";
int nchans = 2048, nsamp = 8192, ncoeffs = 12;
float channel_thresh = 3, spectrum_thresh = 0.5;
unsigned channel_block = 64;

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


// ==========================================================================

// Compute power from input complex values
// A[N] = A[N].x * A[N].x + A[N].y * A[N].y
// Performed in place (data will still consume 32-bits in GPU memory)
__global__ void power(float *data, unsigned nsamp)
{
    for(unsigned s = blockIdx.x * blockDim.x + threadIdx.x; 
                 s < nsamp;
                 s += gridDim.x * blockDim.x)
    {
        short2 value = *((short2 *) &data[s]);
        data[s] = value.x * value.x + value.y * value.y;
    }
}

// Compute the first pass for bandpass generation
// Sum along the channels to get averaged sum, which will be use
// to compute the polynomial co-efficients and fit
__global__ void bandpass_power_sum(float *input, double *bandpass, unsigned nsamp)
{
    // Declare shared memory to store temporary mean and stddev
    __shared__ double local_sum[BANDPASS_THREADS];

    // Initialise shared memory
    local_sum[threadIdx.x] = 0;

    // Synchronise threads
    __syncthreads();

    // Loop over samples
    for(unsigned s = threadIdx.x;
                 s < nsamp; 
                 s += blockDim.x)

        local_sum[threadIdx.x] += input[blockIdx.x * nsamp + s]; 

    // Synchronise threads
    __syncthreads();

    // Use reduction to calculate block mean and stddev
	for (unsigned i = BANDPASS_THREADS / 2; i >= 1; i /= 2)
	{
		if (threadIdx.x < i)
            local_sum[threadIdx.x]  += local_sum[threadIdx.x + i];
		
		__syncthreads();
	}

    // Finally, return temporary sum
    if (threadIdx.x == 0)
        bandpass[blockIdx.x] = local_sum[0] / nsamp;
}

// --------------------- Perform rudimentary RFI clipping: clip channels ------------------------------
__global__ void channel_flagger(float *input, double *bandpass, char *flags, unsigned nsamp, unsigned nchans, 
                                unsigned channel_block, unsigned num_blocks, float channelThresh)
{
    __shared__ float local_mean[BANDPASS_THREADS];

    // 2D Grid, Y-dimension handles channels, X-dimension handles spectra
    float bp_value = __double2float_rz(bandpass[blockIdx.y]);
    local_mean[threadIdx.x] = 0;

    // Loop over all blocks allocated to this threadblock
    for(unsigned b = blockIdx.x; 
                 b < num_blocks; 
                 b += gridDim.x)
    {
        // Load all required value for current block into shared memory
        for(unsigned s = threadIdx.x;
                     s < channel_block;
                     s += blockDim.x)
            local_mean[threadIdx.x] += input[blockIdx.y * nsamp + b * channel_block + s];
        
        __syncthreads();

        // Perform reduction-sum to calculate the mean for current channel block
        for(unsigned i = BANDPASS_THREADS / 2; i > 0; i /= 2)
        {
            if (threadIdx.x < i)
                local_mean[threadIdx.x] += local_mean[threadIdx.x + i];
            __syncthreads();
        }

        // Check if block exceed desired threshold
        if (threadIdx.x == 0 && local_mean[0] / channel_block > bp_value + channelThresh)
        { 
            // Flag current block
            flags[blockIdx.y * num_blocks + b] = 1;

            // Flag neighboring block
            if (b > 0)
                flags[blockIdx.y * num_blocks + b - 1] = 1;
            if (b < num_blocks - 2)
                flags[blockIdx.y * num_blocks + b + 1] = 1;
        }

        // Synchronise threads
        __syncthreads();
    }
}

__global__ void channel_clipper(float *input, double *bandpass, char *flags, unsigned nsamp, unsigned nchans, 
                                unsigned channel_block, unsigned num_blocks)
{
    // 2D Grid, Y-dimension handles channels, X-dimension handles spectra
    float bp_value = __double2float_rz(bandpass[blockIdx.y]);

    // Loop over all channels blocks    
    for (unsigned b = blockIdx.x; 
                  b < num_blocks; 
                  b += gridDim.x)
    {
        // Check if current block is flagged
        if (flags[blockIdx.y * num_blocks + b])
        {
            // This block contains RFI, set to bandpass value
            for(unsigned s = threadIdx.x;
                         s < channel_block;
                         s += blockDim.x)
                input[blockIdx.y * nsamp + b * channel_block + s] -= (input[blockIdx.y * nsamp + blockIdx.x * channel_block + s] - bp_value);
        }
        
    }
}

// --------------------- Perform rudimentary RFI clipping: clip spectra ------------------------------
__global__ void spectrum_clipper(float *input, double *bandpass, unsigned nsamp, 
                                 unsigned nchans, float spectrum_thresh)
{
    // First pass done, on to second step
    // Second pass: Perform wide-band RFI clipping
    for(unsigned s = blockIdx.x * blockDim.x + threadIdx.x;
                 s < nsamp;
                 s += gridDim.x * blockDim.x)
    {
        // For each spectrum, we need to calculate the mean
        float spectrum_mean = 0;

        // All these memory accesses should be coalesced (as thread-spectrum mapping is contiguous)    
        for(unsigned c = 0; c < nchans; c++)
            spectrum_mean += input[c * nsamp + s];
        spectrum_mean /= nchans;

        // We have the spectrum mean, check if it satisfies spectrum threshold
        if (spectrum_mean > spectrum_thresh)
            // Spectrum is RFI, replace with bandpass
            for(unsigned c = 0; c < nchans; c++)
                input[c * nsamp + s] = 512;//__double2float_rz(bandpass[c]);
    }
}

// ============================= CPU BANDPASS FIT ====================================
bool polynomialfit(int obs, int degree, double *dx, double *dy, double *store) /* n, p */
{
    gsl_multifit_linear_workspace *ws;
    gsl_matrix *cov, *X;
    gsl_vector *y, *c;
    double chisq;
 
    int i, j;
 
    X = gsl_matrix_alloc(obs, degree);
    y = gsl_vector_alloc(obs);
    c = gsl_vector_alloc(degree);
    cov = gsl_matrix_alloc(degree, degree);
 
    for(i=0; i < obs; i++) 
    {
        gsl_matrix_set(X, i, 0, 1.0);
        for(j=0; j < degree; j++)
            gsl_matrix_set(X, i, j, pow(dx[i], j));
        gsl_vector_set(y, i, dy[i]);
    }
 
    ws = gsl_multifit_linear_alloc(obs, degree);
    gsl_multifit_linear(X, y, c, cov, &chisq, ws);
 
    /* store result ... */
    for(i=0; i < degree; i++)
        store[i] = gsl_vector_get(c, i);
 
    gsl_multifit_linear_free(ws);
    gsl_matrix_free(X);
    gsl_matrix_free(cov);
    gsl_vector_free(y);
    gsl_vector_free(c);
    return true; // Check the result to know if the fit is good (conv matrix)
}

// Process command-line parameters
void process_arguments(int argc, char *argv[])
{
    int i = 1;
    
    while((fopen(argv[i], "r")) != NULL)
        i++;

    while(i < argc) {
       if (!strcmp(argv[i], "-nchans"))
           nchans = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-nsamp"))
           nsamp = atoi(argv[++i]);
       i++;
    }
}

// Main function
int main(int argc, char *argv[])
{
    unsigned i, j;

    struct timeval start, end;
    long mtime, seconds, useconds; 

	cudaEvent_t event_start, event_stop;
	float timestamp;
	cudaEventCreate(&event_start); 
	cudaEventCreate(&event_stop); 

    // Allocate and initialise CPU and GPU memory for data and bandpass
    float *buffer; double *bandpass, *fitted_bandpass;
    CudaSafeCall(cudaMallocHost((void **) &buffer, nchans * nsamp * sizeof(float), cudaHostAllocPortable));
    CudaSafeCall(cudaMallocHost((void **) &fitted_bandpass, nchans * sizeof(double), cudaHostAllocPortable));
    CudaSafeCall(cudaMallocHost((void **) &bandpass, nchans * sizeof(double), cudaHostAllocPortable));

    // Read data from file and reset initialise bandpass to 0
    read_data(buffer, nsamp, nchans);
    memset(bandpass, 0, nchans * sizeof(double));

    float *d_buffer; double *d_bandpass; char *d_flags;
    cudaMalloc((void **) &d_buffer, nchans * nsamp * sizeof(float));
	cudaMalloc((void **) &d_bandpass, nchans * sizeof(double) );
    cudaMalloc((void **) &d_flags, nchans * (nsamp / channel_block) * sizeof(char));
    cudaMemset(d_bandpass, 0, nchans * sizeof(double));
    cudaMemset(d_flags, 0, (nchans * nsamp / channel_block) * sizeof(char));

    // Copy input buffer to GPU memory
    cudaEventRecord(event_start, 0);
    cudaMemcpy(d_buffer, buffer, nchans * nsamp * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Copied data to GPU in: %lf\n", timestamp);

    // First pass for bandpass fitting, compute sums
    cudaEventRecord(event_start, 0);
    bandpass_power_sum<<<nchans, BANDPASS_THREADS>>>(d_buffer, d_bandpass, nsamp);
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Computed bandpass sum in : %lf\n", timestamp);

    // Second pass, calculate bandpass co-efficients and bandpass fit
    gettimeofday(&start, NULL);

    CudaSafeCall(cudaMemcpy(bandpass, d_bandpass, nchans * sizeof(double), cudaMemcpyDeviceToHost));

    double X[nchans], coeffs[ncoeffs];
    for(i = 0; i < nchans; i++) X[i] = 0 + i / (1.0 * nchans);

    // Fit polynomial using GNU Scientific Library
    polynomialfit(nchans, ncoeffs, X, bandpass, coeffs);

    FILE *fpb = fopen("Test_bandpass_fit.dat", "wb");
    fwrite(bandpass, sizeof(double), nchans, fpb);

    // Generate 1D polynomial using bandpass co-efficients
    // We also need the fit-corrected bandpass to compute the bandpass RMS
    memset(fitted_bandpass, 0, nchans * sizeof(double));
    for(i = 0; i < nchans; i++)
        for(j = 0; j < ncoeffs; j++)
            fitted_bandpass[i] += coeffs[j] * pow(X[i], j);

    fwrite(bandpass, sizeof(double), nchans, fpb);
    fwrite(fitted_bandpass, sizeof(double), nchans, fpb);
    fclose(fpb);

    // Compute bandpass RMS
    float bandpass_mean = 0, bandpass_std = 0;

    // First iteration to compute mean
    for(i = 0; i < nchans; i++)
        bandpass_mean += bandpass[i] / nchans;
    
    // Second iteration, compute standard deviation
    for(i = 0; i < nchans; i++)
        bandpass_std += (bandpass[i] - bandpass_mean) * (bandpass[i] - bandpass_mean);
    bandpass_std = sqrt(bandpass_std / nchans);

    // Calculate the RMSE 
    float rmse = 0;
    for(i = 0; i < nchans; i++)
        rmse += pow(bandpass[i] - fitted_bandpass[i], 2);
    rmse = sqrt(rmse / nchans);

    // Copy bandpass back to GPU memory
    CudaSafeCall(cudaMemcpy(d_bandpass, fitted_bandpass, nchans * sizeof(double), cudaMemcpyHostToDevice));

    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;
    mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
    printf("Calculated Bandpass co-efficients and fit in : %ldms\n", mtime);

    // RFI Clipping, launch GPU-RFI clipper
    channel_thresh  *= rmse;
    spectrum_thresh = bandpass_mean + spectrum_thresh * bandpass_std;

    // We need two passes to perform channel clipping
    // Pass one: mark all blocks which contain RFI
    cudaEventRecord(event_start, 0);
    printf("RMSE: %f, threshold: %f\n", rmse, channel_thresh);
    channel_flagger<<< dim3(nsamp/channel_block, nchans), BANDPASS_THREADS >>>
                   (d_buffer, d_bandpass, d_flags, nsamp, nchans, channel_block, nsamp / channel_block, channel_thresh);

    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Marked channels in : %lf\n", timestamp);

    // Pass two: Set the value with RFI blocks to bandpass value
    cudaEventRecord(event_start, 0);
    channel_clipper<<< dim3(nsamp/channel_block, nchans), BANDPASS_THREADS >>>
                   (d_buffer, d_bandpass, d_flags, nsamp, nchans, channel_block, nsamp / channel_block);

    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Clipped channels in : %lf\n", timestamp);

    cudaEventRecord(event_start, 0);
    printf("Bandpass mean: %f, std: %f, threshold: %f\n", bandpass_mean, bandpass_std, spectrum_thresh);
    spectrum_clipper<<< nsamp / BANDPASS_THREADS, BANDPASS_THREADS >>>
                   (d_buffer, d_bandpass, nsamp, nchans, spectrum_thresh);

    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Clipped spectra in : %lf\n", timestamp);

    // Copy result back to CPU memory
    cudaEventRecord(event_start, 0);
    CudaSafeCall(cudaMemcpy(buffer, d_buffer, nsamp * nchans * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Copied results back to CPU memory in : %lf\n", timestamp);

    // Write buffer to file
    FILE *fp = fopen("Test_bandpass.dat", "wb");
    fwrite(buffer, nchans * nsamp, sizeof(float), fp);
    fclose(fp);
    
}

    
