#ifndef DEDISPERSE_KERNEL_H_
#define DEDISPERSE_KERNEL_H_

#include <cutil_inline.h>

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
// When nbins >= nsamp
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
        for(unsigned j = wingLen + blockIdx.x * blockDim.x + threadIdx.x; 
                     j < fftsize - wingLen;
                     j += blockDim.x * gridDim.x)
        {
            // Calculate input index and output bin
            unsigned bin = (shift + nsamp * i + blockIdx.x * blockDim.x + threadIdx.x) % nbins;
                
            // Calculate power and add to appropriate profile bin
            float2 val = data[blockIdx.y * numBlocks * fftsize + fftsize * i + j];
            profile[blockIdx.y * nbins + bin] += val.x * val.x + val.y * val.y;
        }
    }
}

// When nbins < nsamp
__global__ void detect_smallfold(cufftComplex *data, float *profile, const int nchans, const int fftsize, 
			    const int numBlocks, const int wingLen, float tsamp, unsigned nbins, unsigned shift)
{
    // Check if this thread is a valid profile bin
    if (blockIdx.x * blockDim.x + threadIdx.x >= nbins)
        return;

    // Assign thread a bin
    unsigned binId = (shift + blockIdx.x * blockDim.x + threadIdx.x) % nbins;

    float value = 0;
    unsigned nsamp = fftsize - wingLen * 2;
    for(unsigned i = blockIdx.x * blockDim.x + threadIdx.x; 
                 i < nsamp * numBlocks;
                 i += nbins)
    {
        unsigned index = (i / nsamp) * fftsize + wingLen + i % nsamp;
    
        float2 val = data[i];//blockIdx.y * numBlocks * fftsize + index];
        value += val.x * val.x + val.y * val.y;
    }

    profile[blockIdx.y * nbins + binId] += value;
}

// ---------------------- Fix inter-channel disperion (vector multiply) ------------------------------
__global__ void shift_channels(cufftComplex *data, float obsFreq, float bw, const float dm, 
                               const int bins, const float tsamp)
{
    // Check if this thread points to a valid input bin
    if (blockIdx.x * blockDim.x + threadIdx.x > bins)
        return;

    // Calculate observing and bin frequncy and delay channel
    obsFreq = obsFreq + (bw / 2.0);
    float fchan = obsFreq - (bw / blockDim.y) * blockIdx.y;
    float delay = 4.15e-6 * ((1.0 / (obsFreq-fchan) * (obsFreq-fchan)) - 1.0/(obsFreq*obsFreq) ) 
                  * dm / tsamp;
    float coeffTemp = 2 * M_PI * -delay;

    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
                  i < bins;
                  i += blockDim.x * gridDim.x)
    {
        // Calculate current bin
        unsigned bin = i;
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
    cufftComplex total;
    for (unsigned i = 0; i < nchans; i++ )
    {
        cufftComplex value = data[i * bins + blockIdx.x * blockDim.x + threadIdx.x];
        total.x += value.x;
        total.y += value.y;
    }
        
    // Store total in global memory
    data[blockIdx.x * blockDim.x + threadIdx.x] = total;
}

#endif
