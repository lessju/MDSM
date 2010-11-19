#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "unistd.h"
#include "time.h"
#include "string.h"
#include <cutil_inline.h>

// Stores output value computed in inner loop for each thread
// __shared__ float localvalue[512];

// Stores temporary shift values
__constant__ float dm_shifts[4096];

// Declare texture 
texture<float, 1, cudaReadModeElementType> texRef;

// ---------------------- Optimised Dedispersion Loop  ------------------------------
__global__ void dedisperse_loop(float *outbuff, int nsamp, int nchans, float tsamp,
                                float startdm, float dmstep, int maxshift)
{
    extern __shared__ float localvalue[];

    int c, s, shift;
    float shift_temp = (startdm + blockIdx.y * dmstep) / tsamp;
    
    for(s = threadIdx.x + blockIdx.x * blockDim.x; 
        s < nsamp; 
        s += blockDim.x * gridDim.x) {

           localvalue[threadIdx.x] = 0;
     
           for(c = 0; c < nchans; c++) {
               shift = c * (nsamp + maxshift) + dm_shifts[c] * shift_temp;
//                printf("Tid: %d, bIdx: %d, bIdy: %d, s: %d, c: %d, val: %f \n", threadIdx.x, blockIdx.x, blockIdx.y, s, c, tex2D(texRef, s, c));
                localvalue[threadIdx.x] += tex1D(texRef, shift + s); //s + dm_shifts[c] * shift_temp, c);
           }

           outbuff[blockIdx.y * nsamp + s] = localvalue[threadIdx.x];
       }
}

// -------------------------- Main Program -----------------------------------


float fch1 = 156, foff = -0.005859375, tsamp = 0.000165, dmstep = 0.02, startdm = 0;
int nchans = 1024, nsamp = 1024, tdms = 1024;
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
       else if (!strcmp(argv[i], "-tsamp"))
           blocksize = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-foff"))
           foff = -atof(argv[++i]);
       i++;
    }
}

// Fill buffer with data (blocking call)
void generate_data(float* buffer, int nsamp, int nchans)
{
    for(int i = 0; i < nsamp * nchans; i++)
        buffer[i] = 0.1;
}

// DM delay calculation
float dmdelay(float f1, float f2)
{
  return(4148.741601 * ((1.0 / f1 / f1) - (1.0 / f2 / f2)));
}

int main(int argc, char *argv[])
{
   float *input, *output, *d_output;
   int maxshift, i, j;

   process_arguments(argc, argv);

    // Calculate temporary DM-shifts
    float *dmshifts = (float *) malloc(nchans * sizeof(float));
    for (unsigned i = 0; i < nchans; i++)
          dmshifts[i] = dmdelay(fch1 + (foff * i), fch1);

    // Calculate maxshift
    maxshift = dmshifts[nchans - 1] * (startdm + dmstep * tdms) / tsamp;

    // Allocate and initialise arrays
    input = (float *) malloc( (nsamp + maxshift) * nchans * sizeof(float));
    output = (float *) malloc( nsamp * tdms * sizeof(float));
    for(i = 0; i < nchans; i++)
        for(j = 0; j < nsamp + maxshift; j++)
            input[i * (nsamp + maxshift) + j] = i;

    // Initialise CUDA Events
    cutilSafeCall( cudaSetDevice(0));
    cudaEvent_t event_start, event_stop;
    float timestamp, kernelTime;

    cudaEventCreate(&event_start); 
    cudaEventCreate(&event_stop);

    printf("nsamp: %d, nchans: %d, tsamp: %f, startdm: %f, dmstep: %f, tdms: %d, fch1: %f, foff: %f, maxshift: %d\n",
           nsamp, nchans, tsamp, startdm, dmstep, tdms, fch1, foff, maxshift);

    // Allocate CUDA memory and copy dmshifts
    //cutilSafeCall( cudaMalloc((void **) &d_input, (nsamp + maxshift) * nchans * sizeof(float)));
    cutilSafeCall( cudaMalloc((void **) &d_output, nsamp * tdms * sizeof(float)));
    cutilSafeCall( cudaMemset(d_output, 0, nsamp * tdms * sizeof(float)));
    cutilSafeCall( cudaMemcpyToSymbol(dm_shifts, dmshifts, nchans * sizeof(int)) );

    // Create cuda array on the device
    cudaArray* cuArray;
    cudaMallocArray (&cuArray, &texRef.channelDesc, (nsamp + maxshift) * nchans, 1);
    cudaMemcpyToArray(cuArray, 0, 0, input, sizeof(float) * (nsamp + maxshift) * nchans, cudaMemcpyHostToDevice);
   
    // Bind texture tp the Cuda array
    cudaBindTextureToArray(texRef, cuArray);

    // Set texture attributes
    texRef.normalized = false;
    texRef.filterMode = cudaFilterModePoint;  

    time_t start = time(NULL);

    dim3 gridDim(gridsize, tdms);  
    cudaEventRecord(event_start, 0);
    dedisperse_loop<<<gridDim, blocksize, blocksize>>>(d_output, nsamp, nchans, tsamp, startdm, dmstep, maxshift);
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

    int val = 0;
    for(i = 0; i < nchans; i++) val += i;

    for(i = 0; i < tdms; i++)
        for(j = 0; j < nsamp; j++)
            if (output[i * nsamp + j] != val) {
                printf("Error: dm: %d nsamp: %d value:%f \n", i, j, output[i*nsamp+j]);
                exit(0);
             }
}

