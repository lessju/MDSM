#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cutil_inline.h>

int nchans = 64, nsamp = 4096, ncalls = 1, lo_bin = 64;

// Temp chans store
__shared__ float localvalue[4080];

// --------------------------- Data binning kernel ----------------------------
__global__ void binning_kernel(float *input, int nsamp, int nchans, int binsize, int inshift, int outshift)
{
    int b, c, channel, shift;

    // Loop over all values (nsamp * nchans)
    shift = threadIdx.x + blockIdx.y * (nchans / gridDim.y);
    channel = shift + blockDim.x * gridDim.y * blockIdx.x;
    for(c = 0; c + channel < nsamp * nchans; c += gridDim.x * blockDim.x * gridDim.y * binsize) {

        // Load data from binsize samples into shared memory
        localvalue[threadIdx.x] = 0;

        for(b = 0; b < binsize; b++)
            localvalue[threadIdx.x] += input[inshift + c + blockIdx.x * gridDim.y * 
                                             blockDim.x * binsize + nchans * b + shift];
 
        // Copy data to global memory
        input[outshift + channel + c/binsize] = localvalue[threadIdx.x] / sqrtf(binsize);
    }
}

// --------------------------- In-place data binning kernel ----------------------------
__global__ void inplace_binning_kernel(float *input, int nsamp, int nchans, int binsize)
{
    int b, c, channel, shift;

    // Loop over all values (nsamp * nchans)
    shift = threadIdx.x + blockIdx.y * (nchans / gridDim.y);
    channel = shift + blockIdx.x * gridDim.y * blockDim.x * binsize;
    for(c = 0; c + channel < nsamp * nchans; c += gridDim.x * blockDim.x * gridDim.y * binsize) {

        // Load data from binsize samples into shared memory
        localvalue[shift] = 0;

        for(b = 0; b < binsize; b++)
            localvalue[shift] += input[c + blockIdx.x * gridDim.y * blockDim.x * 
                                       binsize + nchans * b + shift];

        // Copy data to global memory
        input[c +  channel] = localvalue[shift] / binsize;
    }
}

__global__ void inplace_memory_reorganisation(float *input, int nsamp, int nchans, int binsize)
{
    int c, channel, shift;

    // Loop over all values (nsamp * nchans)
    shift = threadIdx.x + blockIdx.y * (nchans / gridDim.y);
    channel = shift + blockDim.x * gridDim.y * blockIdx.x;
    for(c = 0; c + channel < nsamp * nchans; c += gridDim.x * blockDim.x * gridDim.y * binsize) {

        // Load data from binsize samples into shared memory
        localvalue[shift] = input[c + blockIdx.x * gridDim.y * blockDim.x * binsize + shift];
 
        // Copy data to global memory
        input[channel + c/binsize] = localvalue[shift];
    }
}

// --------------------------- Main processing  ----------------------------

// Process command-line parameters
void process_arguments(int argc, char *argv[])
{
    int i = 1;

    while(i < argc) {
       if (!strcmp(argv[i], "-nchans"))
           nchans = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-nsamp"))
           nsamp = atoi(argv[++i]);
       i++;
    }
}

// Fill buffer with data (blocking call)
void generate_data(float* buffer, int nsamp, int nchans)
{
    int i;

    for(i = 0; i < nsamp * nchans; i++)
        buffer[i] = 1;
}


int main(int argc, char *argv[])
{
   float *input, *d_input;
   int i, j;

   process_arguments(argc, argv);
   printf("nsamp: %d, nchans: %d\n", nsamp, nchans);

   // Calculate amount of memory required to do all the downsampling on the GPU
   long long memory = 0, binsize = lo_bin;
   for(i = 0; i < ncalls; i++) {
       memory += (nsamp * nchans) / binsize;
       printf("memory += %d\n", (nsamp * nchans) / binsize );
       binsize *= 2;
   }       
   memory *= sizeof(float);
   memory = max(memory, (long long) (nchans * nsamp * sizeof(float)));

   printf("Memory required: %d\n", memory );

   // Allocate buffers
   input = (float *) malloc(memory * 2);
   generate_data(input, nsamp, nchans);

   // Initialise
   cutilSafeCall( cudaSetDevice(0));

   cudaEvent_t event_start, event_stop;
   float timestamp;
   int gridsize = 64;
   dim3 gridDim(gridsize, nchans / 128.0 < 1 ? 1 : nchans / 128.0);
   dim3 blockDim(min(nchans, 128), 1);
   printf("Grid dimensions: %d x %d, block dimensions: %d x 1\n", gridDim.x, gridDim.y, blockDim.x);

   cudaEventCreate(&event_start); 
   cudaEventCreate(&event_stop); 

   // Allocate GPU memory and copy data
   cutilSafeCall( cudaMalloc((void **) &d_input, memory ));
   cutilSafeCall( cudaMemcpy(d_input, input, nsamp * nchans * sizeof(float), cudaMemcpyHostToDevice) );

   // Call kernel
   cudaEventRecord(event_start, 0);

   int kernelBin = binsize = lo_bin; 
   int inshift = 0, outshift = 0;
   for( i = 0; i < ncalls; i++) {

       if (binsize != 1) {        // if binsize is 1, no need to perform binning
           if (i == 0) {          // Original raw data not required, special case
               printf("(Inplace) Binsize: %d, kernelBin: %d, nsamp: %d, inshift: %d, outshift: %d\n", binsize, kernelBin, nsamp, inshift, outshift);
               inplace_binning_kernel<<< gridDim, blockDim >>>(d_input, nsamp, nchans, kernelBin); 
               inplace_memory_reorganisation<<< gridDim, blockDim >>>(d_input, nsamp, nchans, kernelBin); 
               cutilSafeCall( cudaMemset(d_input + nsamp * nchans / binsize, 0, (nsamp * nchans - nsamp * nchans / binsize) * sizeof(float)));
           } else {
               inshift = outshift;
               outshift += (nsamp * nchans) * 2 / binsize;
               printf("Binsize: %d, kernelBin: %d, nsamp: %d, inshift: %d, outshift: %d\n", binsize, kernelBin, nsamp * 2 / binsize, inshift, outshift);
               binning_kernel<<< gridDim, blockDim >>>(d_input, nsamp * 2 / binsize, nchans, kernelBin, inshift, outshift); 
           }
       }

       binsize *= 2;
       kernelBin = 2;
   }

   cudaEventRecord(event_stop, 0);
   cudaEventSynchronize(event_stop);
   cudaEventElapsedTime(&timestamp, event_start, event_stop);
   printf("Binned data in: %lf\n", timestamp);

    // Get output 
    cutilSafeCall( cudaMemcpy(input, d_input, memory, cudaMemcpyDeviceToHost) );

  //  int total_samp = memory / sizeof(float) / nchans;
    int total_samp = nsamp;
    for(i = 0; i < total_samp; i++) {
        for(j = 0; j < nchans; j++)
           printf("%.0f ", input[i * nchans + j]);
         printf("\n");
        } 

//    int offset = 0, bin = lo_bin;
//    for(i = 0; i < ncalls; i++) {
//        for(j = 0; j < nsamp * nchans / bin; j++)
//            if (input[offset + j] != bin) {
//                printf("Invalid: %d, %d, %d, %d, %.0f\n", i, bin, offset + j, j, input[offset + j]);
//                exit(0);
//            }
//        offset += (nsamp * nchans) / bin;
//        bin *= 2;
//    } 

}
