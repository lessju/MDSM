#ifndef BEAMFORMING_KERNEL_H_
#define BEAMFORMING_KERNEL_H_

#define BEAMFORMER_THREADS 128
#define BEAMS_PER_TB 8
#define BEAMS 8
#define ANTS 32
#define HEAP 128 // Specific to Medicina data output format

// --------------------------------------- Beamforming Kernels --------------------------------------

// Kernel which paralellises over time instead of frequency within the blocks
// Medicina implementation
__constant__ signed char lookup_table[16];

// Kernel which paralellises over time instead of frequency within the blocks
// Medicina implementation... this assumes 32 antennas
__global__ void 
__launch_bounds__(BEAMFORMER_THREADS) 
beamformer(char4 *input, float *output, float2 *shifts, unsigned nsamp, unsigned nchans)
{   
    // Shared memory store for phase shifts
    // The inner-most loop will split antennas into groups of four, so we only need
    // BEAM_PER_TB * 8 floats per iteration
    __shared__ float2 coefficients[BEAMS_PER_TB * 4];

    // Threablock will loop over time for a single channel
    // Groups of beams change in the z-direction
    // Channel changes in the y-direction
    // Multiple blocks in the x-direction

    // Loop over time samples for current block
    for(unsigned time = blockIdx.x * blockDim.x + threadIdx.x;
                 time < nsamp;
                 time += gridDim.x * blockDim.x)
    {
        // Compute index to start of block
        unsigned index = (blockIdx.y * nsamp + blockIdx.x * blockDim.x) * ANTS / 4;

        // Initialise beam registers
        register float beams_real[BEAMS_PER_TB] = { 0 };
        register float beams_imag[BEAMS_PER_TB] = { 0 };

        // Loop over all antennas and compute phase components
        for(unsigned antenna = 0;
                     antenna < ANTS / 4;
                     antenna++)
        {
            // Load antenna values from global memory
            char4 antenna_val = input[index + threadIdx.x * ANTS / 4 + antenna];

            // Load shifts associated with these four antennas and all beams for current thread block
            for(unsigned i = threadIdx.x;
                         i < 4 * BEAMS_PER_TB;
                         i += blockDim.x)
                coefficients[i] = shifts[blockIdx.y * BEAMS * ANTS +
                                         antenna * 4 * BEAMS + blockIdx.z * BEAMS_PER_TB +
                                         i];

            float4 ant_real = {  lookup_table[(antenna_val.w >> 4) & 0xF],  lookup_table[(antenna_val.x >> 4) & 0xF],
                                 lookup_table[(antenna_val.y >> 4) & 0xF],  lookup_table[(antenna_val.z >> 4) & 0xF] };
            float4 ant_imag = {  lookup_table[antenna_val.w & 0x0F],        lookup_table[antenna_val.x & 0x0F],
                                 lookup_table[antenna_val.y & 0x0F],        lookup_table[antenna_val.z & 0x0F] };

            // Synchronise threads
            __syncthreads();

            // Loop over all beams
            for(unsigned beam = 0;
                         beam < BEAMS_PER_TB;
                         beam++)
            {
                float2 shift;

                shift = coefficients[beam];
                beams_real[beam] += ant_real.w * shift.x + ant_imag.w * shift.y;
                beams_imag[beam] += ant_imag.w * shift.x + ant_real.w * shift.y;

                shift = coefficients[BEAMS_PER_TB + beam];
                beams_real[beam] += ant_real.x * shift.x + ant_imag.x * shift.y;
                beams_imag[beam] += ant_imag.x * shift.x + ant_real.x * shift.y;

                shift = coefficients[2 * BEAMS_PER_TB + beam];
                beams_real[beam] += ant_real.y * shift.x + ant_imag.y * shift.y;
                beams_imag[beam] += ant_imag.y * shift.x + ant_real.y * shift.y;

                shift = coefficients[3 * BEAMS_PER_TB + beam];
                beams_real[beam] += ant_real.z * shift.x + ant_imag.z * shift.y;
                beams_imag[beam] += ant_imag.z * shift.x + ant_real.z * shift.y;
            }
        }

        // Add phase and amplitude parts and save computed beams to global memory
        for(unsigned beam = 0; beam < BEAMS_PER_TB; beam++)
            output[(blockIdx.z * BEAMS_PER_TB + beam) * nsamp * nchans + blockIdx.y * nsamp + time] = 
                 __fsqrt_rz(beams_real[beam] * beams_real[beam] + beams_imag[beam] * beams_imag[beam]);

        // Synchronise threads
        __syncthreads();
    }
}

// Kernel which paralellises over time instead of frequency within the blocks
// Medicina implementation... this assumes 32 antennas
// Does not compute the beam power but leaves them complex
__global__ void 
__launch_bounds__(BEAMFORMER_THREADS) 
beamformer_complex(char4 *input, float2 *output, float2 *shifts, unsigned nsamp, unsigned nchans)
{   
    // Shared memory store for phase shifts
    // The inner-most loop will split antennas into groups of four, so we only need
    // BEAM_PER_TB * 8 floats per iteration
    __shared__ float2 coefficients[BEAMS_PER_TB * 4];

    // Threablock will loop over time for a single channel
    // Groups of beams change in the z-direction
    // Channel changes in the y-direction
    // Multiple blocks in the x-direction

    // Loop over time samples for current block
    for(unsigned time = blockIdx.x * blockDim.x + threadIdx.x;
                 time < nsamp;
                 time += gridDim.x * blockDim.x)
    {
        // Compute index to start of block
        unsigned index = (blockIdx.y * nsamp + blockIdx.x * blockDim.x) * ANTS / 4;

        // Initialise beam registers
        register float beams_real[BEAMS_PER_TB] = { 0 };
        register float beams_imag[BEAMS_PER_TB] = { 0 };

        // Loop over all antennas and compute phase components
        for(unsigned antenna = 0;
                     antenna < ANTS / 4;
                     antenna++)
        {
            // Load antenna values from global memory
            char4 antenna_val = input[index + threadIdx.x * ANTS / 4 + antenna];

            // Load shifts associated with these four antennas and all beams for current thread block
            for(unsigned i = threadIdx.x;
                         i < 4 * BEAMS_PER_TB;
                         i += blockDim.x)
                coefficients[i] = shifts[blockIdx.y * BEAMS * ANTS +
                                         antenna * 4 * BEAMS + blockIdx.z * BEAMS_PER_TB +
                                         i];

            float4 ant_real = {  lookup_table[(antenna_val.w >> 4) & 0xF],  lookup_table[(antenna_val.x >> 4) & 0xF],
                                 lookup_table[(antenna_val.y >> 4) & 0xF],  lookup_table[(antenna_val.z >> 4) & 0xF] };
            float4 ant_imag = {  lookup_table[antenna_val.w & 0x0F],        lookup_table[antenna_val.x & 0x0F],
                                 lookup_table[antenna_val.y & 0x0F],        lookup_table[antenna_val.z & 0x0F] };

            // Synchronise threads
            __syncthreads();

            // Loop over all beams
            for(unsigned beam = 0;
                         beam < BEAMS_PER_TB;
                         beam++)
            {
                float2 shift;

                shift = coefficients[beam];
                beams_real[beam] += ant_real.w * shift.x + ant_imag.w * shift.y;
                beams_imag[beam] += ant_imag.w * shift.x + ant_real.w * shift.y;

                shift = coefficients[BEAMS_PER_TB + beam];
                beams_real[beam] += ant_real.x * shift.x + ant_imag.x * shift.y;
                beams_imag[beam] += ant_imag.x * shift.x + ant_real.x * shift.y;

                shift = coefficients[2 * BEAMS_PER_TB + beam];
                beams_real[beam] += ant_real.y * shift.x + ant_imag.y * shift.y;
                beams_imag[beam] += ant_imag.y * shift.x + ant_real.y * shift.y;

                shift = coefficients[3 * BEAMS_PER_TB + beam];
                beams_real[beam] += ant_real.z * shift.x + ant_imag.z * shift.y;
                beams_imag[beam] += ant_imag.z * shift.x + ant_real.z * shift.y;
            }
        }

        // Add phase and amplitude parts and save computed beams to global memory
        for(unsigned beam = 0; beam < BEAMS_PER_TB; beam++)
        {
            unsigned index = (blockIdx.z * BEAMS_PER_TB + beam) * nsamp * nchans + blockIdx.y * nsamp + time;
            output[index].x = beams_real[beam];
            output[index].y = beams_imag[beam];
        }

        // Synchronise threads
        __syncthreads();
    }
}

// ======================= Downsample KernelS =============================
// Downfactor generated beam down to the required sampling time
__global__ void downsample(float *input, float *output, unsigned nsamp, unsigned nchans, unsigned nbeams, unsigned factor)
{
    // Each thread block processes one channel/beam vector

    // Loop over all time samples
    for(unsigned s = 0;
                 s < nsamp / (blockDim.x * factor);
                 s++)
    {
        // Index for thread block starting point
        unsigned index = (blockIdx.y * nchans + blockIdx.x) * nsamp + s * blockDim.x * factor;

        // Perform local downsampling and store to local accumulator
        float value = 0;
        for(unsigned i = 0; i < factor; i++)
            value += input[index + threadIdx.x * factor + i];

        // Output needs to be transposed in memory
         output[(s * blockDim.x + threadIdx.x) * nchans * nbeams + blockIdx.x * nbeams + blockIdx.y] = value;
    }
}

// Downfactor generated beam down to the required sampling time using atomics
// NOTE: factor must be >= 64
__global__ void downsample_atomics(float *input, float *output, unsigned nsamp, unsigned nchans, unsigned nbeams, unsigned factor)
{
    // Each thread block processes one channel/beam vector
    extern __shared__ float vector[];

    // Loop over all time samples
    for(unsigned s = blockIdx.x;
                 s < nsamp / factor;
                 s += gridDim.x)
    {
        // Load input value
        float local_value = input[(blockIdx.z * nchans + blockIdx.y) * nsamp + s * factor + threadIdx.x];

        // Use kepler atomics to perform partial reduction
        local_value += __shfl_down(local_value, 1, 2);
        local_value += __shfl_down(local_value, 2, 4);
        local_value += __shfl_down(local_value, 4, 8);
        local_value += __shfl_down(local_value, 8, 16);
        local_value += __shfl_down(local_value, 16, 32);

        // Synchronise thread to finalise first part of partial reduction
        __syncthreads();

        // Store required value to shared memory
        if (threadIdx.x % 32 == 0)
            vector[threadIdx.x / 32] = local_value;

        // Synchronise thread
        __syncthreads();

        // Perform second part of reduction
        for (unsigned i = factor / 64; i >= 1; i /= 2)
	    {
		    if (threadIdx.x < i)
                vector[threadIdx.x] += vector[threadIdx.x + i];
		
		    __syncthreads();
	    }

        // Store to output (transposed)
        if (threadIdx.x == 0)
            output[s * nchans * nbeams + blockIdx.y * nbeams + blockIdx.z] = vector[0];

        // Synchronise threads
        __syncthreads();
    
    }
}

// ======================= Rearrange Kernel =============================
// Re-organise data after finer channelisation
// No donwsampling performed 
__global__ void fix_channelisation(float2 *input, float *output, unsigned nsamp, unsigned nchans, unsigned nbeams, 
                                   unsigned subchans, unsigned start_chan)
{    
    // Time changes in the x direction
    // Channels change along the y direction. Indexing start at start_chan
    // Beams change along the z direction
    // Each thread processes one sample
    for(unsigned s = blockIdx.x;
                 s < nsamp / subchans;
                 s += gridDim.x)
    {
        // Get index to start of current channelised block
        // ThreadIdx.x is the nth channel formed in this block
        unsigned index = blockIdx.z * nchans * nsamp + (start_chan + blockIdx.y) * nsamp + s * subchans + threadIdx.x;
        float2 value = input[index];

        // Store this in output buffer (transposed)
        index = s * nbeams * gridDim.y * subchans + (blockIdx.y * subchans + threadIdx.x) * nbeams + blockIdx.z;
        output[index] = __fsqrt_rz(value.x * value.x + value.y * value.y);
    }
}

// ======================= Rearrange Kernel =============================
// Rearrange medicina antenna data to match beamformer required input 
// Threadblock size is HEAP]

__global__ void rearrange_medicina(unsigned char *input, unsigned char *output, unsigned nsamp, unsigned nchans)
{
    // Each grid row processes a separate channel
    // Each grid column processes a separate heap
    for(unsigned h = blockIdx.x;
                 h < nsamp / HEAP;
                 h += gridDim.x)
    {
        unsigned int index = blockIdx.y * nsamp * ANTS + h * HEAP * ANTS;
     
        // Thread ID acts as pointer to required sample
        for(unsigned a = 0; a < ANTS * 0.5; a++)
        {
            output[index + threadIdx.x * ANTS + a * 2 + 1] = input[index + a * HEAP * 2 + threadIdx.x * 2 + 1];
            output[index + threadIdx.x * ANTS + a * 2    ] = input[index + a * HEAP * 2 + threadIdx.x * 2];
        }
    }
}
// --------------------------------------------------------------------------------------------------------------------

#endif