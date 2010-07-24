#ifndef DEDISPERSE_KERNEL_H_
#define DEDISPERSE_KERNEL_H_

#include <cutil_inline.h>

//#define FERMI

// Stores temporary shift values
__device__ __constant__ float dm_shifts[16384];

// Stores output value computed in inner loop for each sample
//#ifdef FERMI
//	__device__ __shared__ float localvalue[8192];
//#else
	__device__ __shared__ float localvalue[4008];
//#endif

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
            indx = (soffset + (int) (dm_shifts[c * chanfactor] * shift_temp)) * nchans + c;
            localvalue[threadIdx.x] += buff[inshift + indx];
        }

        outuff[outshift + blockIdx.y * nsamp + soffset] = localvalue[threadIdx.x];
    }
}

// -------------------------- The Subband Dedispersion Loop -----------------------------------
// TODO: Optimise (process each subband in a different thread, instead of a sample per thread)
__global__ void dedisperse_subband(float *outbuff, float *buff, int nsamp, int nchans, int nsubs, 
                                   float startdm, float dmstep, float tsamp, int inshift, int outshift)
{
    int samp, s, c, indx, soffset, sband, tempval, chans_per_sub = nchans / nsubs;
    float shift_temp;

    s = threadIdx.x + blockIdx.x * blockDim.x;
    shift_temp = (startdm + blockIdx.y * dmstep) / tsamp;

    // dedispersing loop over all samples in this buffer
    for (samp = 0; s + samp < nsamp; samp += blockDim.x * gridDim.x) {
        soffset = (s + samp);       

        // loop over the subbands
        for (sband = 0; sband < nsubs; sband++) {  

            // Clear array element for storing dedispersed subband
            localvalue[threadIdx.x * nsubs + sband] = 0.0;

            // Subband channels are shifted to sample location of the highest frequency
            tempval = (int) (dm_shifts[sband * chans_per_sub] * shift_temp); 

            // Add up channels within subband range
            for (c = (sband * chans_per_sub); c < (sband + 1) * chans_per_sub; c++) {
                indx = (soffset + (int) (dm_shifts[c] * shift_temp - tempval)) * nchans + c;
                localvalue[threadIdx.x * nsubs + sband] += buff[inshift + indx];
            }

            // Store values in global memory
            outbuff[outshift + blockIdx.y * nsamp * nsubs + soffset * nsubs + sband] = localvalue[threadIdx.x * nsubs + sband];
        }
    }
}

// ----------------------------- Channel Binnig Kernel --------------------------------
__global__ void channel_binning_kernel(float *input, int nchans, int binsize)
{
    int b, c, channel;
    float total;

    channel = threadIdx.x + blockDim.x * blockIdx.x;
    for(c = 0; c + channel < nchans; c += gridDim.x * blockDim.x) {

        localvalue[threadIdx.x] = input[c + channel];
    
        __syncthreads();
 
        if (threadIdx.x % binsize == 0) {       
            total = 0;
            for(b = 0; b < binsize; b++)       
               total += localvalue[threadIdx.x + b];
            input[c + channel] = total;
        }

       __syncthreads();
    }
}

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
        input[outshift + channel + c/binsize] = localvalue[threadIdx.x] / binsize;// sqrtf(binsize);
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
        input[c +  channel] = localvalue[shift] / binsize;//sqrtf(binsize);
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

// -----------------------------------------------------------------------------------

#endif
