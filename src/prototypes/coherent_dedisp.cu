#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "unistd.h"
#include "time.h"
#include "string.h"
#include <cufft.h>

#include "cpgplot.h"

#define BLOCKSIZE 512

// =========== HELPER FUNCTIONS TO READ IN GUPPI-PSRFITS FILES ================
FILE *fp;
long int seekPos = 0;
void getData(cufftComplex *data, unsigned nchans, unsigned nSamples, unsigned overlap, unsigned counter)
{
    if (counter == 0)
        fp = fopen("PSRDATA/PSRB2021+51_X.dat", "rb");

    // Seek to correct position
    if (fseek(fp, seekPos, SEEK_SET) != 0)
    {
        perror("Could not seek file\n");
        exit(0);
    }
    seekPos += (nSamples - overlap * 2) * nchans * 2 * sizeof(float);

    // Read data segment
    float *buffer = (float *) malloc(nSamples * nchans * 2 * sizeof(float));
    fread(buffer, sizeof(float), nSamples * nchans * 2, fp);

	// Read data whilst transposing
	for (unsigned t = 0; t < nSamples; t++)
    {
		for(unsigned c = 0; c < nchans; c++)
		{
			data[c * nSamples + t].x = buffer[t * nchans * 2 + c * 2];
			data[c * nSamples + t].y = buffer[t * nchans * 2 + c * 2 + 1];
		}
    }

    free(buffer);
}

// ============================================================================

// ---------------------- Coherent Dedispersion - single DM - single Pol ------------------------------
// obsFreq and bw in MHz
// bw in entire band BW
__global__ void coherent_dedisp(cufftComplex *data, float obsFreq, float bw, const float dm, 
                                const int nchans, const int samples, const int fftsize)
{
    // Check if this thread points to a valid input sample
    if (blockIdx.x * blockDim.x + threadIdx.x > samples)
        return;

    // Calculate observing and bin frequncy for current channel/sample
    obsFreq    = obsFreq - (bw / 2) + ((bw / nchans) * blockIdx.y) + bw / (nchans * 2);
    bw         = bw / nchans;  // Convert band bw to channel bw
    float freq = (blockDim.x * blockIdx.x + threadIdx.x) * bw / fftsize;
    freq       = (blockDim.x * blockIdx.x + threadIdx.x >= fftsize / 2) 
                 ? freq - bw
                 : freq;

    // Calculate chirp phase for bin represented by this thread
    cufftComplex chirp;
    const float phase = (bw / abs(bw)) * -2 * M_PI * dm / 2.41033087e-10 *
                        ((freq / obsFreq) * (freq / obsFreq)) / (obsFreq + freq);

    chirp.x = cos(phase) * (1.0 / fftsize);
    chirp.y = -sin(phase) * (1.0 / fftsize);

    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
                  i < samples;
                  i += fftsize)
    {
        // Get input sample from global memory
        cufftComplex value = data[blockIdx.y * samples + i];
    
        // Perform vector multiply
        float2 result;
        result.x = value.x * chirp.x - value.y * chirp.y;
        result.y = value.y * chirp.x + value.x * chirp.y;

        // Save result in global memory
        data[blockIdx.y * samples + i] = result;
    }
}

// ------------------------- Detect and fold -----------------------------
//  device_idata, device_profile, nchans, fftsize, numBlocks, overlap/2, tsamp, nbins, shift
__global__ void detect_fold(cufftComplex *data, float *profile, const int nchans, const int fftsize, 
			    const int numBlocks, const int wingLen, float tsamp, unsigned nbins, unsigned shift)
{
    // Check if this thread points to a valid input sample with an FFT block
    if (wingLen + blockIdx.x * blockDim.x + threadIdx.x >= fftsize - wingLen)
        return;

    // Valid samples in each fftsize (wingLen/2 invalid samples at each buffer end)
    unsigned nsamp = fftsize - wingLen * 2;

    // Loop over all FFT blocks
    for(unsigned i = 0; i < numBlocks; i++)
    {
        // Calculate input index and output bin
        unsigned index = fftsize * i + wingLen + blockIdx.x * blockDim.x + threadIdx.x;
        unsigned bin   = (shift + nsamp * i + blockIdx.x * blockDim.x + threadIdx.x) % nbins;
            
        // Calculate power and add to appropriate profile bin
        float2 val = data[blockIdx.y * numBlocks * fftsize + index];
        profile[blockIdx.y * nbins + bin] += val.x * val.x + val.y * val.y;
    }
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
    // Number of channels, observing frequency and bandwidth in MHz (of whole band)
    unsigned nchans    = 32;
    float    obsFreq   = 1150;
    float    bw        = 100;
    float    dm         = 22;  
    double   tsamp     = 3.2e-7;
    double   period    = 0.529196917808;
    double   currT     = 0;
    unsigned folding   = 1;
    unsigned nbins     = period / tsamp;

    // Calculate required chirp length, overlap size and usable fftsize for convolution
    // chirp_len will consider the lowest frequency channel
    float lofreq = obsFreq - fabs(bw / 2.0);
    float hifreq = obsFreq - fabs(bw / 2.0) + fabs(bw / nchans);

    unsigned chirp_len = 4.150e6 * dm * (pow(lofreq, -2) - pow(hifreq, -2)) * abs(bw * 1e3);

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
    // We define numBlocks... for now
    fftsize /= 2;
    unsigned numBlocks = 2 ;
    unsigned gpuSamples = numBlocks * (fftsize - overlap) + overlap;
    unsigned nsamp = gpuSamples - overlap;

    // ---------------------- Initialise CUDA stuff ---------------------------------
    cudaSetDevice(2);
    cudaEvent_t event_start, event_stop;
    float timestamp;

    // Events
    cudaEventCreate(&event_start); 
    cudaEventCreate(&event_stop);

    // cufft stuff
    cufftHandle plan;
    cufftComplex *device_idata,  *host_idata;
    cufftComplex *host_odata;
    float *device_profile, *host_profile;

    // CUDA memory 
    unsigned inputSize   = gpuSamples * nchans * sizeof(cufftComplex);
    unsigned outputSize  = nsamp * nchans * sizeof(cufftComplex);
    unsigned profileSize = nbins * sizeof(float) * nchans;
    cudaMalloc((void **) &device_idata, fftsize * numBlocks * nchans * sizeof(cufftComplex));
    cudaMalloc((void **) &device_profile, profileSize);
    cudaMemset(device_profile, 0, profileSize);

    printf("\n\tnchans: %d, dm: %f, nsamp: %d, gpuSamples: %d\n"
           "\tchirp_len: %d, overlap: %d, fftsize: %d, numBlocks: %d, nsamp: %d\n"
           "\tGPU buffer size: %.2f MB, Profile size: %.2f MB\n\n",

               nchans, dm, nsamp, gpuSamples,
               chirp_len, overlap, fftsize, numBlocks, nsamp,
               (fftsize * numBlocks * nchans * sizeof(cufftComplex)) / (1024 * 1024.0),
               profileSize / (1024.0 * 1024.0));

    checkCUDAError("Alloacting memory on GPU");
    
    // Allocate host buffers
    host_odata   = (cufftComplex *) malloc(outputSize);
    host_idata   = (cufftComplex *) malloc(inputSize);
    host_profile = (float *) malloc(profileSize);

    // Setup plotter
    if(cpgbeg(0, "/xwin", 1, 1) != 1)
        return EXIT_FAILURE; 

    unsigned size, decFactor;
    if (folding)
    { 
        size = nbins; decFactor = 8192; 
        cpgenv(0.0, size/decFactor, 0, 250*2, 0, 1);
    }
    else
    { 
        size = nsamp; decFactor = 512; 
        cpgenv(0.0, size/decFactor, 0, 150, 0, 1);
    }

    // Create FFT plans
    cufftPlan1d(&plan, fftsize, CUFFT_C2C, nchans * numBlocks);

    // Process multiple buffers
    FILE *fp = fopen("outputChan.dat", "wb");
    for(unsigned counter = 0; counter < 50*4; counter ++)
    {
        // Update timestamp
        currT = (overlap / 2 + counter * nsamp) * tsamp;

        // --------------- Generate/Read Data and reset memory------------------
        memset(host_idata, 0, nchans * gpuSamples * sizeof(cufftComplex));
        memset(host_odata, 0, outputSize);
        cudaMemset(device_idata, 0, fftsize * numBlocks * nchans * sizeof(cufftComplex));

	    getData(host_idata, nchans, gpuSamples, overlap / 2, counter); // Read data from file

        // ---------------------- Copy data to GPU ----------------------------------
        cudaEventRecord(event_start, 0);

        // The +overlap for gpuSamples comes in in the last fftsize copy to GPU memory
        for(unsigned i = 0; i < nchans; i++)
            for(unsigned j = 0; j  < numBlocks; j++)
                cudaMemcpy(device_idata + (i * numBlocks + j) * fftsize, 
                           host_idata + (i * gpuSamples) + j * (fftsize - overlap), 
                           fftsize * sizeof(cufftComplex), 
                           cudaMemcpyHostToDevice);

        cudaEventRecord(event_stop, 0);
        cudaEventSynchronize(event_stop);
        cudaEventElapsedTime(&timestamp, event_start, event_stop);
        checkCUDAError("Copying data to GPU");
        printf("[%d] Copied to GPU in: %lf\n", counter, timestamp);

        // ---------------------- FFT all the channels in place ----------------------
        cudaEventRecord(event_start, 0);
        cufftExecC2C(plan, device_idata, device_idata, CUFFT_FORWARD);
        cudaThreadSynchronize();
        cudaEventRecord(event_stop, 0);
        cudaEventSynchronize(event_stop);
        cudaEventElapsedTime(&timestamp, event_start, event_stop);
        checkCUDAError("Performing FFT");
        printf("Performed forward FFT: %lf\n", timestamp);

        // --------------------- Coherent Dedispersion -------------------------------
	    dim3 gridDim(fftsize / BLOCKSIZE, nchans);
            
	    cudaEventRecord(event_start, 0);
        coherent_dedisp<<<gridDim, BLOCKSIZE >>> (device_idata, obsFreq, bw, dm, nchans, fftsize * numBlocks, fftsize);

        cudaThreadSynchronize();
        cudaEventRecord(event_stop, 0);
        cudaEventSynchronize(event_stop);
        cudaEventElapsedTime(&timestamp, event_start, event_stop);
        checkCUDAError("Performing coherent dedispersion");
        printf("Performed coherent dedispersion: %lf\n", timestamp);

        // --------------------- IFFT channels and DMs in place -----------------------
        cudaEventRecord(event_start, 0);
        cufftExecC2C(plan, device_idata, device_idata, CUFFT_INVERSE);
        cudaThreadSynchronize();
        cudaEventRecord(event_stop, 0);
        cudaEventSynchronize(event_stop);
        cudaEventElapsedTime(&timestamp, event_start, event_stop);
        checkCUDAError("Performing IFFT");
        printf("Performed inverse FFT: %lf\n", timestamp);

        if (folding)
        {
	        // --------------------- Constant-Period folding & power -----------------------
            dim3 gridDim( ceil((fftsize-overlap) / (float) BLOCKSIZE), nchans);
            unsigned shift = fmod(currT, period) / tsamp;

            cudaEventRecord(event_start, 0);
	        detect_fold<<<gridDim, BLOCKSIZE>>>(device_idata, device_profile, nchans, fftsize, 
					                            numBlocks, overlap / 2, tsamp, nbins, shift);

            cudaThreadSynchronize();
            cudaEventRecord(event_stop, 0);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&timestamp, event_start, event_stop);
            checkCUDAError("Detection and Folding");
            printf("Performed detection and folding: %lf\n", timestamp);

            // --------------------- Copy Result back to Host ----------------------------
            cudaEventRecord(event_start, 0);
            cudaMemcpy(host_profile, device_profile, profileSize, cudaMemcpyDeviceToHost);
            cudaEventRecord(event_stop, 0);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&timestamp, event_start, event_stop);
            checkCUDAError("Copying data to host");
            printf("Copied to Host in: %lf [%d]\n", timestamp, shift);

            unsigned plotChannel=17;
            float *xr = (float *) malloc(size/decFactor * sizeof(float));
            float *yr = (float *) malloc(size/decFactor * sizeof(float));
   
            // Decimate before plotting
            for (unsigned i = 0; i < size / decFactor; i++)
            {        
                unsigned index = plotChannel * nbins + i * decFactor;
                xr[i] = i;
                yr[i] = 0;

                for (unsigned j = 0; j < decFactor; j++)
                    yr[i] += host_profile[index + j];

                yr[i] = (yr[i] / decFactor);
            }

            cpgbbuf();
            cpgeras();
            cpgsci(1);
            cpgbox("bcnst", 0, 0.0, "bcnst", 0.0, 0);
            cpgsci(3);
            cpgline(size/decFactor, xr, yr);
            cpgmtxt("T", 2.0, 0.0, 0.0, "Pulsar Profile");
            cpgebuf();
            
            free(xr);
            free(yr);
        }
        else
        {
            // --------------------- Copy Result back to Host ----------------------------
            cudaEventRecord(event_start, 0);

            for(unsigned i = 0; i < nchans; i++)
                for(unsigned j = 0; j < numBlocks; j++)
                    cudaMemcpy(host_odata + i * nsamp + j * (fftsize - overlap),
                               device_idata + i * numBlocks * fftsize + j * fftsize + overlap / 2,
                               (fftsize - overlap) * sizeof(cufftComplex),
                               cudaMemcpyDeviceToHost);

            cudaEventRecord(event_stop, 0);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&timestamp, event_start, event_stop);
            checkCUDAError("Copying data to host");
            printf("Copied to Host in: %lf\n", timestamp);

            float xr[size / decFactor], yr[nchans][size/decFactor];

            for (unsigned c = 0; c < nchans; c++)
            {
                // Decimate before plotting
                for (unsigned i = 0; i < size / decFactor; i++)
                {        
                    unsigned index = c * nsamp + i * decFactor;
                    xr[i] = i;
                    yr[c][i] = 0;

                    for (unsigned j = 0; j < decFactor; j++)
                        yr[c][i] += host_odata[index+j].x * host_odata[index+j].x + 
                                    host_odata[index+j].y * host_odata[index+j].y;

                    yr[c][i] = (yr[c][i] / decFactor) + c * 4;
                }
            }

            fwrite(yr[18], sizeof(float), size / decFactor, fp);
            fflush(fp);

            cpgbbuf();
            cpgeras();
            cpgsci(1);
            cpgbox("bcnst", 0, 0.0, "bcnst", 0.0, 0);
            cpgsci(7);

            for (unsigned i = 0; i < nchans; i++)
                cpgline(size/decFactor, xr, yr[i]);

            cpgmtxt("T", 2.0, 0.0, 0.0, "Dedispersed Channel Plot");
            cpgebuf();
        }
    }   

    cpgend();

    printf("Finished processing \n\n");

}

