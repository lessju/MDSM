#ifndef DEDISPERSE_KERNEL_H_
#define DEDISPERSE_KERNEL_H_

#include "cache_brute_force.h"

// ---------------------- level_one_cache_with_accumulators_brute_force -------------------------------------
//TODO: optimise access to dm_shifts
__global__ void cache_dedispersion(float *output, float *input, float *dm_shifts, 
                                   const int nsamp, const int nchans, const float mstartdm, 
                                   const float mdmstep, const int maxshift)
{
	int   shift;	
	float local_kernel_t[NUMREG];

	int t  = blockIdx.x * NUMREG * blockDim.x  + threadIdx.x;
	
	// Initialise the time accumulators
	for(int i = 0; i < NUMREG; i++) local_kernel_t[i] = 0.0f;

	float shift_temp = mstartdm + ((blockIdx.y * blockDim.y + threadIdx.y) * mdmstep);
	
	// Loop over the frequency channels.
    for(int c = 0; c < nchans; c++) 
    {
		// Calculate the initial shift for this given frequency
		// channel (c) at the current despersion measure (dm) 
		// ** dm is constant for this thread!!**
		shift = (c * (nsamp + maxshift) + t) + __float2int_rz (dm_shifts[c] * shift_temp);
		
        #pragma unroll
		for(int i = 0; i < NUMREG; i++) {
			local_kernel_t[i] += input[shift + (i * DIVINT) ];
		}
	}

	// Write the accumulators to the output array. 
    #pragma unroll
	for(int i = 0; i < NUMREG; i++) {
		output[((blockIdx.y * DIVINDM) + threadIdx.y)* nsamp + (i * DIVINT) + (NUMREG * DIVINT * blockIdx.x) + threadIdx.x] = local_kernel_t[i];
	}
}

// -------------------- 1D Median Filter -----------------------
__global__ void median_filter(float *input, const int nsamp)
{
    // Declare shared memory array to hold local kernel samples
    // Should be (blockDim.x+width floats)
    __shared__ float local_kernel[MEDIAN_THREADS + MEDIAN_WIDTH - 1];

    // Loop over sample blocks
    for(unsigned s = threadIdx.x + blockIdx.x * blockDim.x; 
                 s < nsamp; 
                 s += blockDim.x * gridDim.x)
    {
        // Value associated with thread
        unsigned index = blockIdx.y * nsamp + s;
        unsigned wing  = MEDIAN_WIDTH / 2;

        // Load sample associated with thread into shared memory
        local_kernel[threadIdx.x + wing] = input[index];

        // Synchronise all threads        
        __syncthreads();

        // Load kernel wings into shared memory, handling boundary conditions
        // (for first and last wing elements in time series)
        if (s >= wing && s < nsamp - wing)
        {
            // Not in boundary, choose threads at the edge and load wings
            if (threadIdx.x < wing)   // Load wing element at the beginning of the kernel
                local_kernel[threadIdx.x] = input[blockIdx.y * nsamp + blockIdx.x * blockDim.x - (wing - threadIdx.x)];
            else if (threadIdx.x >= blockDim.x - wing)  // Load wing elements at the end of the kernel
                local_kernel[threadIdx.x + MEDIAN_WIDTH - 1] = input[index + wing];
        }

        // Handle boundary conditions (ignore end of buffer for now)
        else if (s < wing && threadIdx.x < wing + 1)   
            // Dealing with the first items in the input array
            local_kernel[threadIdx.x] = local_kernel[wing];
        else if (s > nsamp - wing && threadIdx.x == blockDim.x / 2)
            // Dealing with last items in the input array
            for(unsigned i = 0; i < wing; i++)
                local_kernel[MEDIAN_THREADS + wing + i] = local_kernel[nsamp - 1];

        // Synchronise all threads and start processing
        __syncthreads();

        // Load value to local registers median using "moving window" in shared memory 
        // to avoid bank conflicts
        float median[MEDIAN_WIDTH];
        for(unsigned i = 0; i < MEDIAN_WIDTH; i++)
            median[i] = local_kernel[threadIdx.x + i];

        // Perform partial-sorting on median array
        for(unsigned i = 0; i < wing + 1; i++)    
            for(unsigned j = i; j < MEDIAN_WIDTH; j++)
                if (median[j] < median[i])
                    { float tmp = median[i]; median[i] = median[j]; median[j] = tmp; }

        // We have our median, store to global memory
        input[index] = median[wing];
    }
}

// ------------------------- Calculate Mean and Standard Deviation ---------------------------
__global__ void mean_stddev(float *input, float2 *stddev, const int nsamp)
{
    // Declare shared memory to store temporary mean and stddev
    __shared__ float local_mean[MEAN_NUM_THREADS];
    __shared__ float local_stddev[MEAN_NUM_THREADS];

    // Initialise shared memory
    local_stddev[threadIdx.x] = 0;
    local_mean[threadIdx.x]   = 0;

    // Synchronise threads
    __syncthreads();

    // Loop over samples
    for(unsigned s = threadIdx.x + blockIdx.x * blockDim.x; 
                 s < nsamp; 
                 s += blockDim.x * gridDim.x)
    {
        float val = input[s];
        local_stddev[threadIdx.x] += (val * val);
        local_mean[threadIdx.x]   += val; 
    }

    // Synchronise threads
    __syncthreads();

    // Use reduction to calculate block mean and stddev
	for (unsigned i = MEAN_NUM_THREADS / 2; i >= 1; i /= 2)
	{
		if (threadIdx.x < i)
		{
            local_stddev[threadIdx.x] += local_stddev[threadIdx.x + i];
            local_mean[threadIdx.x]   += local_mean[threadIdx.x + i];
		}
		
		__syncthreads();
	}

    // Finally, return temporary standard deviation value
    if (threadIdx.x == 0)
    {
        float2 vals = { local_mean[0], local_stddev[0] };
        stddev[blockIdx.x] = vals;
    }
}

// --------------------------------- Detrend and Normalisation kernel ---------------------------------
__global__ void detrend_normalise(float *input, int detrendLen)
{	
	// Store temporary least-square fit values
	__shared__ float3 shared[BANDPASS_THREADS];

	// First pass, each thread computes its part of the buffer
	{
		float sy = 0, sxy = 0, sxx = 0;
		for (unsigned i = threadIdx.x; i < detrendLen; i += blockDim.x)
		{
			float x = - detrendLen / 2.0 + 0.5 + i;
			int index = blockIdx.y * gridDim.x * detrendLen + 
						blockIdx.x * detrendLen + i;
			float y = input[index];

			sy += y;
			sxy += x * y;
			sxx += x * x;
		}

		// Initialise shared memory
		shared[threadIdx.x].x = sy;
		shared[threadIdx.x].y = sxy;
		shared[threadIdx.x].z = sxx;
	}

	__syncthreads();

	// Perform the rest of the computation through reduction
	for (unsigned i = BANDPASS_THREADS / 2; i >= 1; i /= 2)
	{
		if (threadIdx.x < i)
		{
			shared[threadIdx.x].x += shared[threadIdx.x + i].x;
			shared[threadIdx.x].y += shared[threadIdx.x + i].y;
			shared[threadIdx.x].z += shared[threadIdx.x + i].z;
		}
		
		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		shared[0].y /= shared[0].z;
		shared[0].x /= detrendLen;
	}

	__syncthreads();
	
	// Detrend and compute partial standard deviation
	{
		float a = shared[0].x;
		float b = shared[0].y;
		float stddev = 0;

		for (unsigned i = threadIdx.x; i < detrendLen ; i += blockDim.x)
		{
			float x = - detrendLen / 2.0 + 0.5 + i;
			int index = blockIdx.y * gridDim.x * detrendLen + 
						blockIdx.x * detrendLen + i;
			float val = input[index] - (a + b * x);
			input[index] = val;
			stddev += val * val;
		}

		shared[threadIdx.x].z = stddev;
	}

	__syncthreads();

	// Compute the full standard deviation through reduction
	for (unsigned i = BANDPASS_THREADS / 2; i >= 1; i /= 2)
		if (threadIdx.x < i)
			shared[threadIdx.x].z += shared[threadIdx.x + i].z;

	__syncthreads();

	if (threadIdx.x == 0)
		shared[0].z = sqrt(shared[0].z / detrendLen);

	__syncthreads();

	// Normalise Data
	float stddev = shared[0].z;

	for (unsigned i = threadIdx.x; i < detrendLen ; i += blockDim.x)
		input[blockIdx.y * gridDim.x * detrendLen + 
			  blockIdx.x * detrendLen + i] /= stddev;
}

// -------------------------------------- BANDPASS ------------------------------------------

// Compute power from input complex values
// A[N] = A[N].x * A[N].x + A[N].y * A[N].y
// Performed in place (data will still consume 32-bits in GPU memory)
__global__ void voltage_to_power(float *data, unsigned nchans, unsigned shift, unsigned samples, unsigned total)
{
    for(unsigned c = 0; c < nchans; c++)
        for(unsigned s = blockIdx.x * blockDim.x + threadIdx.x; 
                     s < samples;
                     s += gridDim.x * blockDim.x)
        {
            short2 value = *((short2 *) &(data[c * total + shift + s]));
            data[c * total + shift + s] = value.x * value.x + value.y * value.y;
        }
}

// Compute the first pass for bandpass generation
// Sum along the channels to get averaged sum, which will be use
// to compute the polynomial co-efficients and fit
__global__ void bandpass_power_sum(float *input, double *bandpass, unsigned shift, unsigned nsamp, unsigned total)
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

        local_sum[threadIdx.x] += input[blockIdx.x * total + shift + s]; 

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
__global__ void channel_clipper(float *input, double *bandpass, float bandpass_mean, unsigned channel_block, unsigned nsamp, 
                                unsigned nchans, unsigned shift, unsigned total, float channelThresh)
{
    __shared__ float local_mean[BANDPASS_THREADS];

    if (blockIdx.x * blockDim.x + threadIdx.x > nsamp)
        return;

    // 2D Grid, Y-dimension handles channels, X-dimension handles spectra
    float bp_value = __double2float_rz(bandpass[blockIdx.y]);
    local_mean[threadIdx.x] = 0;

    // Load all required values to shared memory
    for(unsigned s = threadIdx.x; 
                 s < channel_block; 
                 s += blockDim.x)
        local_mean[threadIdx.x] += input[blockIdx.y * total + blockIdx.x * channel_block + shift + s] - bp_value;

    __syncthreads();

    // Perform reduction-sum to calculate the mean for current channel block
    for(unsigned i = BANDPASS_THREADS / 2; i >= 1; i /= 2)
    {
        if (threadIdx.x < i)
            local_mean[threadIdx.x] += local_mean[threadIdx.x + i];
        __syncthreads();
    }

    // Compute channel mean
    if (threadIdx.x == 0)
        local_mean[0] /= channel_block;

    __syncthreads();

    // This should be handled as a shared-memory broadcast, check
    // Check if channel block exceeds acceptable limit, if not flag
    if (blockIdx.y >= 85 && blockIdx.y <= 95)
    {
        for(unsigned s = threadIdx.x; 
                     s < channel_block; 
                     s += blockDim.x)
            input[blockIdx.y * total + blockIdx.x * channel_block + shift + s] = bp_value; 
        return;
    }

    if (local_mean[0] > channelThresh)
        for(unsigned s = threadIdx.x; 
                     s < channel_block; 
                     s += blockDim.x)
        input[blockIdx.y * total + blockIdx.x * channel_block + shift + s] -= 
                (input[blockIdx.y * total + blockIdx.x * channel_block + shift + s] - bp_value);

}

// --------------------- Perform rudimentary RFI clipping: clip spectra ------------------------------
__global__ void spectrum_clipper(float *input, double *bandpass, float bandpass_mean, unsigned nsamp, 
                                 unsigned nchans, unsigned shift, unsigned total, float spectrumThresh)
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
            spectrum_mean += (input[c * total + shift + s] - __double2float_rz(bandpass[c]));
        spectrum_mean /= nchans;

        // We have the spectrum mean, check if it satisfies spectrum threshold
        if (spectrum_mean > spectrumThresh)
            // Spectrum is RFI, clear (replace with bandpass for now)
            for(unsigned c = 0; c < nchans; c++)
                input[c * total + shift + s] = __double2float_rz(bandpass[c]);
    }
}

#endif
