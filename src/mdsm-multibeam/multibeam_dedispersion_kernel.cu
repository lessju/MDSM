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

// ---------------------- Shared memory optimised dedisperison  -------------------------------------
__global__ void shared_dedispersion(const float* __restrict__ input, float* __restrict__ output, 
							        const int* __restrict__ all_delays, const unsigned nchans, 
                                    const unsigned nsamp, const int maxshift, const int tdms)
{
		// Shared memory buffer to store channel vector
	extern __shared__ float vector[];

	// Each thread will process a number of DM values associated with one time sample
	register float accumulators[DEDISP_DMS];

	// Initialise shared memory store for dispersion delays
	__shared__ int delays[DEDISP_DMS];

	// Initialise accumulators
	for(unsigned d = 0; d < DEDISP_DMS; d++) accumulators[d] = 0;

	// Loop over all frequency channels
	for(unsigned c = 0; c < nchans; c++)
	{
		// Synchronise threads before updating dispersion delays
		__syncthreads();

		// Load all the shifts associated with this threadblock DM-range for the current channel
		int inshift = all_delays[c * tdms + blockIdx.y * DEDISP_DMS];
		if (threadIdx.x < DEDISP_DMS)
			delays[threadIdx.x] = all_delays[c * tdms + blockIdx.y * DEDISP_DMS + threadIdx.x] - inshift;
		
		// Synchronise threads
		__syncthreads();

		// We'll need to load the channel vector (which will be larger than threadDim
		// due to dispersion
		for(unsigned s = threadIdx.x; 
					 s < blockDim.x + delays[DEDISP_DMS - 1]; 
					 s += blockDim.x)
			vector[s] = input[(maxshift + nsamp) * c + blockIdx.x * blockDim.x + inshift + s];

		// Synchronise threads
		__syncthreads();

		// Loop over DM values associated with current threadblock and update accumulators
		// Manual unlooping of four to overlap shared memory requests
		#pragma unroll
		for(int d = 0; d < DEDISP_DMS; d ++)
            accumulators[d]  += vector[threadIdx.x + delays[d]];
//		{
//			int shift1          = delays[d];
//			int shift2          = delays[d + 1];
//			int shift3          = delays[d + 2];
//			int shift4          = delays[d + 3];
//     		accumulators[d]     += vector[threadIdx.x + shift1];
//			accumulators[d + 1] += vector[threadIdx.x + shift2];
//			accumulators[d + 2] += vector[threadIdx.x + shift3];
//			accumulators[d + 3] += vector[threadIdx.x + shift4];
//		}
	}

	// All done, store result to global memory
    #pragma unroll
	for(unsigned d = 0; d < DEDISP_DMS; d++)
		output[(blockIdx.y * DEDISP_DMS + d) * nsamp + blockIdx.x * blockDim.x + threadIdx.x] = accumulators[d];
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
__global__ void channel_flagger(float *input, double *bandpass, char *flags, unsigned nsamp, unsigned nchans, 
                                unsigned channel_block, unsigned num_blocks, float channelThresh,
                                unsigned total, unsigned shift )
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
            local_mean[threadIdx.x] += input[blockIdx.y * total + b * channel_block + s + shift];
        
        __syncthreads();

        // Perform reduction-sum to calculate the mean for current channel block
        for(unsigned i = BANDPASS_THREADS / 2; i > 0; i /= 2)
        {
            if (threadIdx.x < i)
                local_mean[threadIdx.x] += local_mean[threadIdx.x + i];
            __syncthreads();
        }

        // Check if block exceed desired threshold
        if (threadIdx.x == 0 && (local_mean[0] / channel_block) > bp_value + channelThresh)
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
                                unsigned channel_block, unsigned num_blocks, unsigned total, unsigned shift)
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
                input[blockIdx.y * total + b * channel_block + s + shift] = bp_value;
        }
        
    }
}

// --------------------- Perform rudimentary RFI clipping: clip spectra ------------------------------
__global__ void spectrum_clipper(float *input, double *bandpass, float bandpass_mean, unsigned nsamp, 
                                 unsigned nchans, unsigned shift, unsigned total, float spectrumThresh, float spectrumThreshLow)
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
            spectrum_mean += input[c * total + shift + s];
        spectrum_mean /= nchans;

        // We have the spectrum mean, check if it satisfies spectrum threshold
        if (spectrum_mean > spectrumThresh || spectrum_mean < spectrumThreshLow )
            // Spectrum is RFI, clear (replace with bandpass for now)
            for(unsigned c = 0; c < nchans; c++)
                input[c * total + shift + s] = __double2float_rz(bandpass[c]);
    }
}

// --------------------------------------- Beamforming Kernel --------------------------------------

// Kernel which paralellises over time instead of frequency within the blocks
// Medicina implementation... this assumes 32 antennas
__global__ void 
__launch_bounds__(BEAMFORMER_THREADS) 
beamform_medicina(char4 *input, float *output, float *shifts, unsigned nsamp, unsigned nchans, unsigned output_shift)
{
    // Shared memory store for imaginary components (BEAMFORMER_THREADS * 8 [*4 for char4 implied])   
    __shared__ char4 shared[BEAMFORMER_THREADS * 8];

    // Shared memory store for combined real components (one combined value per thread)
    __shared__ float real[BEAMFORMER_THREADS];

    // Shared memory store for phase shifts
    // The inner-most loop will split antennas into groups of four, so we only need
    // BEAM_PER_TB * 4 float per iteration
    __shared__ float phase_shifts[BEAMS_PER_TB * 4];

    // Threablock will loop over time for a single channel
    // Groups of beams change in the z-direction
    // Channel changes in the y-direction
    // Multiple blocks in the x-direction
    
    // Loop over time samples for current block
    for(unsigned time = blockIdx.x * blockDim.x + threadIdx.x;
                 time < nsamp;
                 time += gridDim.x * blockDim.x)
    {
        // Load antenna data for current spectra subset (BEAMFORMER_THREADS in all) to shared
        // memory, placing phase components in shared and combined amplitudes in real

        // Compute index to start of block
        unsigned index = (blockIdx.y * nsamp + blockIdx.x) * ANTS / 4;
        
        // Loop over antennas in groups of four
        for(unsigned i = threadIdx.x;
                     i < blockDim.x * 8; // One memory request will load 4 full antennas
                     i += blockDim.x)
        {
            // Grab 4 antennas from global memory. This a quarter warp will contain all the antenna values
            // TODO: Check if this is correct
            register char4 value = input[index + i];

            // Each warp make up a single spectrum. Combine the amplitude components and store in real
            float4 real_value = { (value.w >> 4) & 0xF, (value.x >> 4) & 0xF, 
                                  (value.y >> 4) & 0xF, (value.z >> 4) & 0xF };

            // Combine antennas belonging to the current thread
            float   amplitude = real_value.w * real_value.w + real_value.x * real_value.x +
                                real_value.y * real_value.y + real_value.z * real_value.z;

            
            // Use warp-shuffle to combine antennas from 8 different threads
        //    amplitude += __shfl_down(amplitude, 1, 2);
        //    amplitude += __shfl_down(amplitude, 2, 4);
        //    amplitude += __shfl_down(amplitude, 4, 8);

            // Each 8th thread will contain a valid ampltiude value. Store this to real
            if (i % 8 == 0)
                real[i / 8] = amplitude;

            // We are ready from processing the amplitude value. Next we just need to store the
            // phase components to shared
            value.w = value.w & 0x0F;
            value.x = value.x & 0x0F;
            value.y = value.y & 0x0F;
            value.z = value.z & 0x0F;

            shared[i] = value;
        }        

        // Finished pre-computation. Synchronise threads
        __syncthreads();

        // Initialise beam registers
        register float beams[BEAMS_PER_TB] = { 0 };

        // Loop over all antennas and compute phase components
        for(unsigned antenna = 0;
                     antenna < ANTS / 4;
                     antenna++)
        {
            // Add four antennas at a time (to reduce shared memory overhead and increase arithmetic intensity)
            char4 imag_char = shared[threadIdx.x * 8 + antenna];

            float imagw = imag_char.w;
            float imagx = imag_char.x;
            float imagy = imag_char.y;
            float imagz = imag_char.z;

            // Load shifts associated with these four antennas and all beams for current thread block
            for(unsigned i = threadIdx.x; 
                         i < 4 * BEAMS_PER_TB; 
                         i += blockDim.x)
                phase_shifts[i] = shifts[blockIdx.y * BEAMS * ANTS + 
                                         antenna * 4 * BEAMS + blockIdx.z * BEAMS_PER_TB + i];

            // Synchronise threads
            __syncthreads();
            
            // Loop over all beams 
            for(unsigned beam = 0;
                         beam < BEAMS_PER_TB;
                         beam++)
            {
                // Read shifts from shared memory and apply to current four antennas
				float shift; 

				shift        = phase_shifts[beam];
                beams[beam] += (shift * imagw) * (shift * imagw);
				shift        = phase_shifts[BEAMS_PER_TB + beam];
                beams[beam] += (shift * imagx) * (shift * imagx);
			    shift        = phase_shifts[2 * BEAMS_PER_TB + beam];
                beams[beam] += (shift * imagy) * (shift * imagy);
				shift        = phase_shifts[3 * BEAMS_PER_TB + beam];
                beams[beam] += (shift * imagz) * (shift * imagz);
            }
        }

        // Add phase and amplitude parts and save computed beams to global memory
        for(unsigned beam = 0; beam < BEAMS_PER_TB; beam++)
            output[(blockIdx.z * BEAMS_PER_TB + beam) * nsamp * nchans + blockIdx.y * nsamp + output_shift + time] = 
                    beams[beam] + real[threadIdx.x];

        // Synchronise threads
        __syncthreads();
    }
}

#endif
