#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "unistd.h"
#include "time.h"
#include "string.h"
#include <cufft.h>

#include "cpgplot.h"

#define BLOCKSIZE 256

// =========== HELPER FUNCTIONS TO READ IN GUPPI-PSRFITS FILES ================

void getData(cufftComplex *data, unsigned nchans, unsigned nSamples)
{
    FILE *fp = fopen("PSRDATA/PSRB2021+51_Y.dat", "rb");
	
	float datumX, datumY;

	// Read data whilst transposing
	for (unsigned t = 0; t < nSamples; t++)
		for(unsigned c = 0; c < nchans; c++)
		{
			fread(&datumX, sizeof(float), 1, fp);
			fseek(fp, 1, SEEK_CUR);
			fread(&datumY, sizeof(float), 1, fp);
			fseek(fp, 1, SEEK_CUR);

			data[c * nSamples + t].x = datumX;
			data[c * nSamples + t].y = datumY;
		}

	printf("%f %f %f %f %f %f\n", data[0].x, data[0].y, data[1].x, data[1].y, data[2].x, data[2].y);
}

// ============================================================================


/* General Notes
   -------------

   - Won't work terribly well for a large DM range... multiple iterations by splitting DM range?
   - Special kernel for a single DM value?
*/

// ---------------------- Coherent Dedispersion  ------------------------------
__global__ void coherent_dedisp(cufftComplex *input, cufftComplex *output, 
                                float obsFreq, float bw, const float startDM, const float dmStep,
                                const int nchans, const int nsamp, const int ndms)
{
    // Check if this thread points to a valid input sample
    if (blockIdx.x * blockDim.x + threadIdx.x > nsamp)
        return;

    // Shared memory to store input samples for reuse
    __shared__ cufftComplex values[BLOCKSIZE];

    // Calculate chirp coefficient once
    const float coeff = (bw / abs(bw)) * 2 * M_PI / 2.41e-10;

    // Read thread data value into shared memory
    values[threadIdx.x] = input[blockIdx.y * nsamp      + 
                                blockIdx.x * blockDim.x + 
                                threadIdx.x];

    // Synchronise threads after writing to shared memory
    __syncthreads();

    // Calculate observing and bin frequncy for current channel/sample
    obsFreq           = obsFreq - (bw / 2) + ((bw / nchans) * blockIdx.y);
    const float freq  = (blockIdx.x * blockDim.x + threadIdx.x)    * 
                        ((bw / nchans) / nsamp) - 0.5 * (bw / nchans);

    // Loop over all DM values
    for (unsigned i = 0; i < ndms; i++)   
    {
        // Calculate chirp phase for current bin
        const float phase = coeff * (startDM + i * dmStep) * obsFreq * obsFreq * 
                                    freq * freq / (obsFreq + freq);

        cufftComplex value = values[threadIdx.x];
        cufftComplex chirp;

        chirp.x =  cos(phase) * (1.0 / nsamp);
        chirp.x = -sin(phase) * (1.0 / nsamp);

        // Vector multiply
        float2 result;
        result.x = value.x * chirp.x - value.y * chirp.y;
        result.y = value.y * chirp.x + value.x * chirp.y;

        // Save result in global memory
        output[i * nchans * nsamp      + 
               blockIdx.y * nsamp      + 
               blockIdx.x * blockDim.x + 
               threadIdx.x]            = result;
    }
   
}

// ----------------------  Detection and Summation Kernel  ------------------------------
__global__ void detection_summation_intensity(cufftComplex *input, const int nchans, 
                                              const int nsamp, const int ndms, const int npols)
{
    // Check if this thread points to a valid input sample
    if (blockIdx.x * blockDim.x + threadIdx.x > nsamp)
        return;

    // Loop over all channels
    float result = 0;

    // Loop over all polarisations
    for (unsigned p = 0; p < npols; p++)
        for (unsigned i = 0; i < nchans; i++) 
        {
            unsigned index = p * ndms * nchans * nsamp   +
                             blockIdx.y * nchans * nsamp + 
                             i * nsamp                   + 
                             blockIdx.x * blockDim.x     + 
                             threadIdx.x;

            float2 val  = input[index];
            result     += val.x * val.x + val.y * val.y;        
        }

    input[blockIdx.y * nchans * nsamp + 
          blockIdx.x * blockDim.x     + 
          threadIdx.x].x              = result;
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

// -------------------------- Main Program ----------------------------------- //
int main()
{
    unsigned i, j;
    
    // Dedispersion range and step
    float dmStart = 22;  
    float dmStep  = 1;
    unsigned ndms = 1;

    // Number of channels, observing frequency and bandwidth in MHz (of whole band)
    unsigned npols     = 1;
    unsigned nchans    = 32;
    float    obsFreq   = 1148.7375;
    float    bw        = 100 / 32.0;

    // Calculate required chirp length, overlap size and usable fftsize for convolution
    // chirp_len will consider the largest DM value at the lowest frequency channel
    float lofreq = obsFreq - abs(bw / 2.0);
    float hifreq = obsFreq - abs(bw / 2.0) + abs(bw / nchans);

    unsigned chirp_len = 4.150e6 * (dmStart + dmStep * ndms)           * 
                                   (pow(lofreq, -2) - pow(hifreq, -2)) * 
                                    abs(bw * 1e3);

    // Set overlap as the next power of two from the chirp len
    unsigned overlap   = pow(2, ceil(log2((float) chirp_len)));

    // Calculate an efficient fftsize (multiple simulataneous FFTs will be computed)
    // NOTE: Stolen from GUPPI dedisperse_gpu.cu

    unsigned fftsize = 16 * 1024;
    if      (overlap <= 1024)    fftsize = 32 * 1024;
    else if (overlap <= 2048)    fftsize = 64 * 1024;
    else if (overlap <= 16*1024) fftsize = 128 * 1024;
    else if (overlap <= 64*1024) fftsize = 256 * 1024;

    while (fftsize < 2 * overlap) fftsize *= 2;
    
    if (fftsize > 2 * 1024 * 1024) {
        printf("FFT length is too large, cannot dedisperse\n");
        exit(0);
    }

    // Calculate number of gpu samples which can be processed in the GPU, and calculate
    // number of input samples required for this (excluding the buffer edge wings)
    unsigned gpuSamples = 1024 * 1024;  // This needs to be calculated properly depding on input parameters
    unsigned numBlocks  = gpuSamples / fftsize;
    unsigned buffLen    = numBlocks * (fftsize - overlap);

    // ---------------------- Initialise CUDA stuff ---------------------------------
    cudaSetDevice(0);
    cudaEvent_t event_start, event_stop;
    float timestamp;

    // Events
    cudaEventCreate(&event_start); 
    cudaEventCreate(&event_stop);

    // cufft stuff
    cufftHandle plan;
    cufftComplex *device_idata, *host_idata;
    cufftComplex *device_odata, *host_odata;

    // CUDA memory
    unsigned outputSize = gpuSamples * npols * ndms * nchans * sizeof(cufftComplex);
    unsigned inputSize  = gpuSamples * npols * nchans * sizeof(cufftComplex);
    cudaMalloc((void **) &device_idata, inputSize);
    cudaMalloc((void **) &device_odata, outputSize);

    printf("\n\tnchans: %d, ndms: %d, nsamp: %d, npols: %d\n"
              "\tchirp_len: %d, overlap: %d, fftsize: %d, numBlocks: %d, gpuSamples: %d\n"
              "\tinputSize: %.2f MB, outputSize: %.2f MB\n\n",

               nchans, ndms, buffLen, npols, 
               chirp_len, overlap, fftsize, numBlocks, gpuSamples,
               inputSize  / (1024 * 1024.0), outputSize / (1024 * 1024.0));

    checkCUDAError("Alloacting memory on GPU");

    // ---------------------- Generate/Read Data ---------------------------------
    host_odata = (cufftComplex *) malloc(outputSize);
    host_idata = (cufftComplex *) malloc(inputSize);
    for(i = 0; i < buffLen * nchans * npols; i++) 
    {
        host_idata[i].x = 1.0f;
        host_idata[i].y = 1.0f;
    }

//	getData(host_idata, nchans, gpuSamples); // Read data from file
	printf("Read / Generated data\n");

    // ---------------------- Copy data to GPU ----------------------------------
    cudaEventRecord(event_start, 0);

    for(unsigned i = 0; i < npols * nchans * numBlocks; i++)
        cudaMemcpy(device_idata + i * fftsize, 
                   host_idata + i * (fftsize - overlap), 
                   fftsize * sizeof(cufftComplex), 
                   cudaMemcpyHostToDevice);

    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    checkCUDAError("Copying data to GPU");
    printf("Copied to GPU in: %lf\n", timestamp);

    // ---------------------- FFT all the channels in place ----------------------
    cufftPlan1d(&plan, fftsize, CUFFT_C2C, npols * nchans * numBlocks);

    cudaEventRecord(event_start, 0);
    cufftExecC2C(plan, device_idata, device_idata, CUFFT_FORWARD);
    cudaThreadSynchronize();
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    checkCUDAError("Performing FFT");
    printf("Performed forward FFT: %lf\n", timestamp);

    // --------------------- Coherent Dedispersion -------------------------------
    dim3 gridDim(gpuSamples / BLOCKSIZE, nchans);

    cudaEventRecord(event_start, 0);

    // Call coherent dedispersion kernel for all beams / polarisations
    for(i = 0; i < npols; i++)
    {
        unsigned shift = i * nchans * gpuSamples;
        coherent_dedisp<<<gridDim, BLOCKSIZE >>> (device_idata + shift, device_odata + shift, obsFreq, 
                                                  bw, dmStart, dmStep, nchans, gpuSamples, ndms);
    }   

    cudaThreadSynchronize();
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    checkCUDAError("Performing coherent dedispersion");
    printf("Performed coherent dedispersion: %lf\n", timestamp);

    // --------------------- IFFT channels and DMs in place -----------------------
    cufftPlan1d(&plan, fftsize, CUFFT_C2C, ndms * nchans * npols * numBlocks);

    cudaEventRecord(event_start, 0);
    cufftExecC2C(plan, device_odata, device_odata, CUFFT_INVERSE);
    cudaThreadSynchronize();
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    checkCUDAError("Performing IFFT");
    printf("Performed inverse FFT: %lf\n", timestamp);

    // --------------------- Detection and Summation -------------------------------
    gridDim.x = gpuSamples / BLOCKSIZE;
    gridDim.y = ndms;

    cudaEventRecord(event_start, 0);
    detection_summation_intensity<<<gridDim, BLOCKSIZE >>> 
              (device_odata, nchans, gpuSamples, ndms, npols);

    cudaThreadSynchronize();
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    checkCUDAError("Performing detection and summation");
    printf("Performed detection and summation: %lf\n", timestamp);

    // --------------------- Copy Result back to Host ----------------------------
    cudaEventRecord(event_start, 0);

    float *output = (float *) host_odata;
//    for (i = 0; i < ndms; i++)
//        for (j = 0; j < numBlocks; j++)
//            cudaMemcpy(output + i * buffLen + j * (fftsize - overlap),
//                       device_odata + i * nchans * fftsize + j * fftsize + overlap / 2,
//                       fftsize - overlap,
//                       cudaMemcpyDeviceToHost);
        
    cudaMemcpy(output, device_odata, outputSize / nchans, cudaMemcpyDeviceToHost);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    checkCUDAError("Copying data to host");
    printf("Copied to Host in: %lf\n", timestamp);

	// Let's try to plot stuff to screen
    if(cpgbeg(0, "?", 1, 1) != 1)
        return EXIT_FAILURE;

	float min = 0, max = 9999999;
	float x[fftsize];
	for(i = 0; i < fftsize; i++)
	{
		if (output[i] < min) min = output[i];
		if (output[i] > max) max = output[i];
		x[i] = i;
	}

    cpgenv(0.0, fftsize, min, max, 0, 1);
    cpglab("x", "y", "Coherent Dedispersion output");
	cpgline(fftsize, x, output);

	cpgend();
    printf("Finished processing \n\n");

}

