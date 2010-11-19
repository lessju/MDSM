#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "unistd.h"
#include "time.h"
#include "string.h"
#include <cutil_inline.h>

// Stores output value computed in inner loop for each thread
__shared__ float localvalue[4008];

// Stores temporary shift values
__constant__ float dm_shifts[1024];

// -------------------------- The Dedispersion Loop -----------------------------------
__global__ void dedisperse_loop(float *outuff, float *buff, int nsamp, int nchans, float tsamp,
                                int chanfactor, float startdm, float dmstep, int inshift, int outshift)
{
    int samp, s, c, indx, soffset;
    float shift_temp;

    /* dedispersing loop over all samples in this buffer */
    s = threadIdx.x + blockIdx.x * blockDim.x;
    shift_temp = (startdm + blockIdx.y * dmstep) / tsamp; 

    for (samp = 0; s + samp < nsamp; samp += blockDim.x * gridDim.x) {
        soffset = (s + samp);
        
        /* clear array element for storing dedispersed subband */
        localvalue[threadIdx.x] = 0.0;

        /* loop over the channels */
        for (c = 0; c < nchans; c ++) {
            indx = (soffset + (int)(dm_shifts[c * chanfactor] * shift_temp)) * nchans + c;
            localvalue[threadIdx.x] += buff[inshift + indx];
        }

        outuff[outshift + blockIdx.y * nsamp + soffset] = localvalue[threadIdx.x];
    }
}

// -------------------------- Main Program -----------------------------------


float fch1 = 126, foff = -6, tsamp = 5e-6, dmstep = 0.065, startdm = 0;
int nchans = 128, nsamp = 1024, tdms = 128;
int gridsize = 128, blocksize = 128;

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
       else if (!strcmp(argv[i], "-dmstep"))
           dmstep = atof(argv[++i]);
       else if (!strcmp(argv[i], "-startdm"))
           startdm = atof(argv[++i]);
       else if (!strcmp(argv[i], "-tdms"))
           tdms = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-gridsize"))
           gridsize = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-blocksize"))
           blocksize = atoi(argv[++i]);
       i++;
    }

    foff = foff / (float) nchans;
    tsamp = tsamp * nchans;
}

// Fill buffer with data (blocking call)
void generate_data(float* buffer, int nsamp, int nchans)
{
    for(int i = 0; i < nsamp * nchans; i++)
        buffer[i] = 01;
}

// DM delay calculation
float dmdelay(float f1, float f2)
{
  return(4148.741601 * ((1.0 / f1 / f1) - (1.0 / f2 / f2)));
}

int main(int argc, char *argv[])
{
   float *input, *output, *d_input, *d_output;
   int maxshift;

   process_arguments(argc, argv);

   printf("nsamp: %d, nchans: %d, tsamp: %f, startdm: %f, dmstep: %f, tdms: %d, fch1: %f, foff: %f\n",
           nsamp, nchans, tsamp, startdm, dmstep, tdms, fch1, foff);

    // Calculate temporary DM-shifts
    float *dmshifts = (float *) malloc(nchans * sizeof(float));
    for (unsigned i = 0; i < nchans; i++)
          dmshifts[i] = dmdelay(fch1 + (foff * i), fch1);

    // Calculate maxshift
    maxshift = dmshifts[nchans - 1] * (startdm + dmstep * tdms) / tsamp;

    // Initialise input buffer
    input = (float *) malloc( (nsamp + maxshift) * nchans * sizeof(float));
    output = (float *) malloc( nsamp * tdms * sizeof(float));

    // Allocate arrays
    input = (float *) malloc( (nsamp + maxshift) * nchans * sizeof(float));
    output = (float *) malloc( nsamp * tdms * sizeof(float));

    // Initialise CUDA stuff
    cutilSafeCall( cudaSetDevice(0));
    cudaEvent_t event_start, event_stop;
    float timestamp, kernelTime;
    dim3 gridDim(gridsize, tdms);  

    cudaEventCreate(&event_start); 
    cudaEventCreate(&event_stop);

    // Allocate CUDA memory and copy dmshifts
    cutilSafeCall( cudaMalloc((void **) &d_input, (nsamp + maxshift) * nchans * sizeof(float)));
    cutilSafeCall( cudaMalloc((void **) &d_output, nsamp * tdms * sizeof(float)));
    cutilSafeCall( cudaMemset(d_output, 0, nsamp * tdms * sizeof(float)));
    cutilSafeCall( cudaMemcpyToSymbol(dm_shifts, dmshifts, nchans * sizeof(int)) );

    time_t start = time(NULL);

    // Copy input to GPU
    cudaEventRecord(event_start, 0);
    cutilSafeCall( cudaMemcpy(d_input, input, (nsamp + maxshift) * nchans * sizeof(float), cudaMemcpyHostToDevice) );    
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied to GPU in: %lf\n", timestamp);

    // Dedisperse
    cudaEventRecord(event_start, 0);
    dedisperse_loop<<<gridDim, blocksize >>>(d_output, d_input, nsamp, nchans, tsamp, 1, startdm, dmstep, 0, 0);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Processed in: %lf\n", timestamp);
    kernelTime = timestamp;

    // Copy output from GPU
    cudaEventRecord(event_start, 0);
    cutilSafeCall( cudaMemcpy(output, d_output, nsamp * tdms * sizeof(float), cudaMemcpyDeviceToHost) );    
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied from GPU in: %lf\n", timestamp);

    printf("Total time: %d\n", (int) (time(NULL) - start));
    printf("Performance: %lf Gflops\n", (nchans * tdms) * (nsamp * 1.0 / kernelTime / 1.0e6));
}

