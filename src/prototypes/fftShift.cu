#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "unistd.h"
#include "time.h"
#include "string.h"
#include <cufft.h>

#include "cpgplot.h"

#define BLOCKSIZE 512

// ---------------------- Fix inter-channel disperion (vector multiply) ------------------------------
// FCH1 = center frequency of top channel
// BW = Total band width
__global__ void shift_channels(cufftComplex *data, float fch1, float bw, const float dm, 
                               const int bins, const float tsamp)
{
    // Check if this thread points to a valid input bin
    if (blockIdx.x * blockDim.x + threadIdx.x > bins)
        return;

    // Calculate observing and bin frequncy and delay channel
    float fchan = fch1 - (bw / gridDim.y) * blockIdx.y;
    float delay = 4148.741601 * ((1.0 / (fchan*fchan)) - 1.0/(fch1*fch1)) * dm / tsamp;
    float coeffTemp = 2 * M_PI * delay * (1.0 / bins);

    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
                  i < bins;
                  i += blockDim.x * gridDim.x)
    {
        // Calculate current bin
        int bin = i;
        if (bin >= floor(bins / 2.0))
            bin -= bins;
        
        cufftComplex coeff;
        coeff.x =  cos(coeffTemp * bin) * (1.0 / bins);
        coeff.y = -sin(coeffTemp * bin) * (1.0 / bins);

        // Get input sample from global memory
        cufftComplex value = data[blockIdx.y * bins + i];
    
        // Perform vector multiply
        float2 result;
        result.x = value.x * coeff.x - value.y * coeff.y;
        result.y = value.y * coeff.x + value.x * coeff.y;

        if (i == bins / 2 && bins % 2 == 0)
            result.y = 0;

        // Save result in global memory
        data[blockIdx.y * bins + i] = result;
    }
}

__global__ void sum_channels(cufftComplex *data, const int bins, const int nchans)
{
    // Check if this thread points to a valid input bin
    if (blockIdx.x * blockDim.x + threadIdx.x > bins)
        return;

    // Loop over all channels and sum up coefficients  
    cufftComplex total = {0, 0};
    for (unsigned i = 0; i < nchans; i++ )
    {
        cufftComplex value = data[i * bins + blockIdx.x * blockDim.x + threadIdx.x];
        total.x += value.x;
        total.y += value.y;
    }
        
    // Store total in global memory
    data[blockIdx.x * blockDim.x + threadIdx.x] = total;
}

// -------------------------- Error checking function ------------------------
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "CUDA error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}

// DM delay helper function
float dmDelay(float dm, float fhi, float flo)
{
    return 4148.741601 * (1.0/(fhi*fhi) - (1.0/(flo*flo))) * dm;
}

// ---------------------------------- main -----------------------------------------

int main()
{
    // Number of channels, observing frequency and bandwidth in MHz (of whole band)
    unsigned N      = 1024 * 32;
    unsigned nchans = 32;
    float    fch1   = 1200;
    float    foff   = 100.0 / nchans;
    float    dm     = 200;  
    float    tsamp  = 5.12e-5;

    // ---------------------- Initialise CUDA stuff ---------------------------------
    cudaSetDevice(0);
    cudaEvent_t event_start, event_stop;
    float timestamp;

    // Events
    cudaEventCreate(&event_start); 
    cudaEventCreate(&event_stop);

    // Create FFT plans
    cufftHandle profile_fplan, profile_iplan;
    cufftPlan1d(&profile_fplan, N, CUFFT_R2C, nchans);
    cufftPlan1d(&profile_iplan, N, CUFFT_C2R, nchans);

    // CUDA memory 
    cufftComplex *device_tempData;
    float        *host_idata, *device_idata, *host_odata, *host_tdata;

    unsigned inputSize   = N * nchans * sizeof(float);
    unsigned tempSize    = N * nchans * sizeof(cufftComplex);
    unsigned outputSize  = N * sizeof(float);

    cudaMalloc((void **) &device_idata, inputSize);
    cudaMalloc((void **) &device_tempData, tempSize);
    cudaMemset(device_idata, 0, inputSize);
    cudaMemset(device_tempData, 0, tempSize);

    checkCUDAError("Alloacting memory on GPU");
    
    // Allocate host buffers
    host_odata   = (float *) malloc(outputSize);
    host_idata   = (float *) malloc(inputSize);
    host_tdata   = (float *) malloc(inputSize);

    // Generate Data
    srand(time(NULL));
    memset(host_idata, 0, inputSize);
    for (unsigned i = 0; i < nchans; i++)
    {
        float delay = dmDelay(dm, fch1 - foff * i - foff / 2, fch1 - foff / 2) / tsamp;
        for(unsigned j = N * 0.25 + delay; 
                     j < N * 0.25 + delay + N * 0.01; 
                     j++)
            host_idata[i * N + j] = 6;
    }

    for(unsigned i = 0; i < nchans * N; i++)
        host_idata[i] += ((double) rand() / (double) RAND_MAX) + 1;

    unsigned blocksize = 512;

    // ---------------------- Copy data to GPU ----------------------------------
    cudaEventRecord(event_start, 0);
    cudaMemcpy(device_idata, host_idata, N * nchans * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    checkCUDAError("Copying data to GPU");
    printf("Copied to GPU in: %lf\n", timestamp);

    // ---------------------- FFT all the channels in place ----------------------
    cudaEventRecord(event_start, 0);
    cufftExecR2C(profile_fplan, device_idata, device_tempData);
    cudaThreadSynchronize();
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    checkCUDAError("Performing FFT");
    printf("Performed forward FFT: %lf\n", timestamp);

    // ------------- Vector multiply with shift components  ------------
    dim3 gridDim(N / blocksize + 1, nchans);

    shift_channels<<<gridDim, blocksize>>>(device_tempData, 
                     fch1 - foff / 2.0, 
                     foff * nchans, dm, N, tsamp);

//    dim3 sumDim(N / blocksize, 1);
//    sum_channels<<< sumDim, blocksize >>>(device_tempData, N, nchans);
//    cudaThreadSynchronize();
//    cudaEventRecord(event_stop, 0);
//    cudaEventSynchronize(event_stop);
//    cudaEventElapsedTime(&timestamp, event_start, event_stop);
//    checkCUDAError("Shifted and summed channels");
//    printf("Shifted and summed channels: %lf\n",  timestamp);

    // ------------- Inverse FFT channel 1 containing full profile  ------------
    cudaEventRecord(event_start, 0);
    cufftExecC2R(profile_iplan, device_tempData, device_idata);
    cudaThreadSynchronize();
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    checkCUDAError("Performing profile IFFT");
    printf("Performed profile IFFT: %lf\n", timestamp);

    // ------------- Copy back to CPU ------------
//    cudaEventRecord(event_start, 0);
//    cudaMemcpy(host_odata, device_idata, outputSize, cudaMemcpyDeviceToHost);
//    cudaEventRecord(event_stop, 0);
//    cudaEventSynchronize(event_stop);
//    cudaEventElapsedTime(&timestamp, event_start, event_stop);
//    checkCUDAError("Copying data from GPU");
//    printf("Copied from GPU in: %lf\n", timestamp);

// ------------------------------------------Unsummed check --------------------
    cudaEventRecord(event_start, 0);
    cudaMemcpy(host_tdata, device_idata, inputSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    checkCUDAError("Copying data from GPU");
    printf("Copied from GPU in: %lf\n", timestamp);

    if(cpgbeg(0, "/xwin", 1, 1) != 1)
        return EXIT_FAILURE; 

    float minY = 9e12, maxY = 9e-12;
    float xr[N], yr[N], zr[N];
    unsigned plotChannel = 8;
    for (unsigned i = 0; i < N; i++)
    {
        xr[i] = i; 
        yr[i] = host_tdata[plotChannel * N + i]; 
        zr[i] = host_idata[plotChannel * N + i];

        if (minY > yr[i]) minY = yr[i];
        if (maxY < yr[i]) maxY = yr[i];
    }

    printf("min: %f, max : %f\n", minY, maxY);

    cpgenv(0.0, N, minY, maxY, 0, 1);
    cpgsci(7);
    cpgline(N, xr, yr);
    cpgsci(6);
    cpgline(N, xr, zr);
    cpgmtxt("T", 2.0, 0.0, 0.0, "Shifted channel");
    cpgend();

// ---------------------------------------------------   // no IFFT check

//    cufftComplex *test = (cufftComplex *) malloc(tempSize);
//    cudaEventRecord(event_start, 0);
//    cudaMemcpy(test, device_tempData, tempSize, cudaMemcpyDeviceToHost);
//    cudaEventRecord(event_stop, 0);
//    cudaEventSynchronize(event_stop);
//    cudaEventElapsedTime(&timestamp, event_start, event_stop);
//    checkCUDAError("Copying data from GPU");
//    printf("Copied from GPU in: %lf\n", timestamp);

//    if(cpgbeg(0, "/xwin", 1, 1) != 1)
//        return EXIT_FAILURE; 

//    float minY = 9e12, maxY = 9e-12;
//    float xr[N], yr[N];
//    unsigned plotChannel = 1;
//    for (unsigned i = 0; i < N; i++)
//    {
//        xr[i] = i; yr[i] = test[plotChannel * N + i].x;
//        if (minY > yr[i]) minY = yr[i];
//        if (maxY < yr[i]) maxY = yr[i];
//    }
////    float xr[nchans], yr[nchans];
////    for(unsigned i = 0; i < nchans; i++)
////    {
////        xr[i] = i; yr[i] = test[i * N].x;
////        if (minY > yr[i]) minY = yr[i];
////        if (maxY < yr[i]) maxY = yr[i];
////    }

//    printf("min: %f, max : %f\n", minY, maxY);

//    cpgenv(0.0, N, minY, maxY, 0, 1);
//    cpgsci(7);
//    cpgline(N, xr, yr);
//    cpgmtxt("T", 2.0, 0.0, 0.0, "Summed profile");
//    cpgend();

// -----------------------------------------------   // Setup plotter
//    if(cpgbeg(0, "/xwin", 1, 1) != 1)
//        return EXIT_FAILURE; 

//    float minY = 9e12, maxY = 9e-12;
//    float xr[N];
//    for (unsigned i = 0; i < N; i++)
//    {
//        xr[i] = i;
//        if (minY > host_odata[i]) minY = host_odata[i];
//        if (maxY < host_odata[i]) maxY = host_odata[i];
//    }

//    cpgenv(0.0, N, minY, maxY, 0, 1);
//    cpgsci(7);
//    cpgline(N, xr, host_odata);
//    cpgmtxt("T", 2.0, 0.0, 0.0, "Summed profile");
//    cpgend();

    printf("Finished processing \n\n");
}

