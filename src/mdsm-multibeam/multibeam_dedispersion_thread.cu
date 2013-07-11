#include "multibeam_dedispersion_kernel.cu"
#include "multibeam_dedispersion_thread.h"
#include "cache_brute_force.h"
#include <gsl/gsl_multifit.h>
#include "math.h"

// PGPLOT
#include "cpgplot.h"

// ===================== CUDA HELPER FUNCTIONS ==========================

// Error checking function
#define CUDA_ERROR_CHECK
#define CudaSafeCall( err ) _cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    _cudaCheckError( __FILE__, __LINE__ )

inline void _cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void _cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

// List devices and assign to process
DEVICES* initialise_devices(SURVEY* survey)
{
	int num_devices;

    // Enumerate devices and create DEVICE_INFO list, storing device capabilities
    CudaSafeCall(cudaGetDeviceCount(&num_devices));

    if (num_devices <= 0)
        { fprintf(stderr, "No CUDA-capable device found\n"); exit(0); }

    // Create and populate devices object
    DEVICES* devices = (DEVICES *) malloc(sizeof(DEVICES));
    devices -> devices = (DEVICE_INFO *) malloc(num_devices * sizeof(DEVICE_INFO));
    devices -> num_devices = 0;
    devices -> minTotalGlobalMem = (1024 * 1024 * 16);

    int orig_num = num_devices, counter = 0;
    char useDevice = 0;
    for(int i = 0; i < orig_num; i++) 
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        useDevice = 0;
        
        if (deviceProp.major == 9999 && deviceProp.minor == 9999)
            { fprintf(stderr, "No CUDA-capable device found\n"); exit(0); }
        else {

            // Check if device is in user specfied list, if any
            if (survey -> gpu_ids != NULL) {
                for(unsigned j = 0; j < survey -> num_gpus; j++)
                    if ((survey -> gpu_ids)[j] == i)
                        useDevice = 1;
            }
            else
                useDevice = 1;

            if (useDevice) 
            {
	            (devices -> devices)[counter].multiprocessor_count = deviceProp.multiProcessorCount;
	            (devices -> devices)[counter].constant_memory = deviceProp.totalConstMem;
	            (devices -> devices)[counter].shared_memory = deviceProp.sharedMemPerBlock;
	            (devices -> devices)[counter].register_count = deviceProp.regsPerBlock;
	            (devices -> devices)[counter].thread_count = deviceProp.maxThreadsPerBlock;
	            (devices -> devices)[counter].clock_rate = deviceProp.clockRate;
	            (devices -> devices)[counter].device_id = i;

	            if (deviceProp.totalGlobalMem / 1024 < devices -> minTotalGlobalMem)
		            devices -> minTotalGlobalMem = deviceProp.totalGlobalMem / 1024;

	            counter++;
                (devices -> num_devices)++;
            }
        }
    }

    if (devices -> num_devices == 0) 
        { fprintf(stderr, "No CUDA-capable device found\n"); exit(0); }

    return devices;
}

// Allocate memory-pinned input buffer
void allocateInputBuffer(float **pointer, size_t size)
{  CudaSafeCall(cudaHostAlloc((void **) pointer, size, cudaHostAllocPortable));  }

// Allocate memory-pinned output buffer
void allocateOutputBuffer(float **pointer, size_t size)
{ CudaSafeCall(cudaHostAlloc((void **) pointer, size, cudaHostAllocPortable)); }


// =================================== BANDPASS FITTING ====================================
bool polynomialfit(int obs, int degree, double *dx, double *dy, double *store) /* n, p */
{
    gsl_multifit_linear_workspace *ws;
    gsl_matrix *cov, *X;
    gsl_vector *y, *c;
    double chisq;
 
    int i, j;
 
    X = gsl_matrix_alloc(obs, degree);
    y = gsl_vector_alloc(obs);
    c = gsl_vector_alloc(degree);
    cov = gsl_matrix_alloc(degree, degree);
 
    for(i=0; i < obs; i++) 
    {
        gsl_matrix_set(X, i, 0, 1.0);
        for(j=0; j < degree; j++)
            gsl_matrix_set(X, i, j, pow(dx[i], j));
        gsl_vector_set(y, i, dy[i]);
    }
 
    ws = gsl_multifit_linear_alloc(obs, degree);
    gsl_multifit_linear(X, y, c, cov, &chisq, ws);

    // Store result
    for(i=0; i < degree; i++)
        store[i] = gsl_vector_get(c, i);
 
    gsl_multifit_linear_free(ws);
    gsl_matrix_free(X);
    gsl_matrix_free(cov);
    gsl_vector_free(y);
    gsl_vector_free(c);
    return true; // Check the result to know if the fit is good (conv matrix)
}

// =================================== CUDA KERNEL HELPERS ====================================

// Calculate signal power from input complex voltages
void calculate_power(float *d_input, THREAD_PARAMS* params, cudaEvent_t event_start, cudaEvent_t event_stop,
                     unsigned shift, unsigned samples, unsigned total)
{
    SURVEY *survey = params -> survey;
    float timestamp;

    cudaEventRecord(event_start, 0);
    voltage_to_power<<< ceil(samples / BANDPASS_THREADS), BANDPASS_THREADS>>>(d_input, survey -> nchans, shift, samples, total);
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("%d. Computed voltage power [Beam %d]: %lf\n", (int) (time(NULL) - params -> start), params -> thread_num, timestamp);
}

// Cache-optimised brute force dedispersion algorithm on the GPU
void cached_brute_force(float *d_input, float *d_output, float *d_dmshifts, THREAD_PARAMS* params, 
                        cudaEvent_t event_start, cudaEvent_t event_stop, unsigned nsamp, int maxshift)
{
    SURVEY *survey = params -> survey;

    int num_reg         = NUMREG;
    int divisions_in_t  = DIVINT;
    int divisions_in_dm = DIVINDM;
    int num_blocks_t    = nsamp / (divisions_in_t * num_reg);
    int num_blocks_dm   = survey -> tdms / divisions_in_dm;

    float timestamp;       
    dim3 threads_per_block(divisions_in_t, divisions_in_dm);
    dim3 num_blocks(num_blocks_t,num_blocks_dm); 

    cudaEventRecord(event_start, 0);	

    cache_dedispersion<<< num_blocks, threads_per_block >>>
                      (d_output, d_input, d_dmshifts, nsamp, 
                       survey -> nchans, survey -> lowdm / survey -> tsamp, 
                       survey -> dmstep/survey -> tsamp, maxshift);

    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("%d: Performed Brute-Force Dedispersion [Beam %d]: %lf\n", 
            (int) (time(NULL) - params -> start), params -> thread_num, timestamp);
}

// Shared memory optimised brute force dedispersion algorithm on the GPU
void shared_brute_force(float *d_input, float *d_output, int *d_all_shifts, THREAD_PARAMS* params, 
                        cudaEvent_t event_start, cudaEvent_t event_stop, unsigned nsamp, int maxshift, unsigned shared_shift)
{
    SURVEY *survey = params -> survey;
    float timestamp;  

    cudaEventRecord(event_start, 0);	

	dim3 gridDim(ceil(nsamp / (1.0 * DEDISP_THREADS)), ceil(survey -> tdms / (1.0 * DEDISP_DMS)));  
    shared_dedispersion<<<gridDim, DEDISP_THREADS, (DEDISP_THREADS + shared_shift) * sizeof(float)>>>
            (d_input, d_output, d_all_shifts, survey -> nchans, nsamp, maxshift, survey -> tdms);

    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("%d: Performed [Shared] Brute-Force Dedispersion [Beam %d]: %lf\n", 
            (int) (time(NULL) - params -> start), params -> thread_num, timestamp);
}

// Calculate bandpass and store relevant information on CPU/GPU
void bandpass_fitting(float *d_input, double *bandpass, double *d_bandpass, THREAD_PARAMS* params, 
                      cudaEvent_t event_start, cudaEvent_t event_stop, unsigned shift, unsigned nsamp,
                      unsigned total)
{
    SURVEY *survey = params -> survey;
    float timestamp;       

    cudaEventRecord(event_start, 0);	

    // First part, calculate summed-bandpass on GPU
    bandpass_power_sum<<<survey -> nchans, BANDPASS_THREADS>>>(d_input, d_bandpass, shift, nsamp, total);
    cudaThreadSynchronize();

    // Get summed-bandpass from GPU
    CudaSafeCall(cudaMemcpy(bandpass, d_bandpass, survey -> nchans * sizeof(double),    
                 cudaMemcpyDeviceToHost));

    // Cacluate X-dimensions (0 -> 1)
    double X[survey -> nchans], coeffs[survey -> ncoeffs];
    for(unsigned i = 0; i < survey -> nchans; i++) X[i] = 0 + i / (1.0 * survey -> nchans);

    // Mask any unwanted channels from bandpass
    for(unsigned i = 0; i < survey -> num_masks; i++)
    {
        unsigned from = (survey -> channel_mask[i]).from, to = (survey -> channel_mask[i]).to;
        float value;
        if (from == 0)
            value = (bandpass[to + 1] + bandpass[to + 2]) * 0.5;
        else if (to >= survey -> nchans - 1)
            value = (bandpass[from - 1] + bandpass[from - 2]) * 0.5;
        else
            value = (bandpass[from - 1] + bandpass[to + 1]) * 0.5;

        for(unsigned i = from; i <= to; i++)
            bandpass[i] = value;
    }

    // Fit polynomial using GNU Scientific Library
    polynomialfit(survey -> nchans, survey -> ncoeffs, X, bandpass, coeffs); 

    // Generate 1D polynomial using bandpass co-efficients
    double fitted_bandpass[survey -> nchans];
    memset(fitted_bandpass, 0, survey -> nchans * sizeof(double)); 

    for(unsigned i = 0; i < survey -> nchans; i++)
        for(unsigned j = 0; j < survey -> ncoeffs; j++)
            fitted_bandpass[i] += coeffs[j] * pow(X[i], j);

    // Asynchronous copy of bandpass
    CudaSafeCall(cudaMemcpyAsync(d_bandpass, fitted_bandpass, survey -> nchans * sizeof(double), 
                 cudaMemcpyHostToDevice));

    // Calculate bandpass statistics to be used later on    
    float bandpass_mean = 0, bandpass_std = 0;

    // First iteration to compute mean
    for(unsigned i = 0; i < survey -> nchans; i++)
        bandpass_mean += bandpass[i] / survey -> nchans;
    
    // Second iteration, compute standard deviation
    for(unsigned i = 0; i < survey -> nchans; i++)
        bandpass_std += (bandpass[i] - bandpass_mean) * (bandpass[i] - bandpass_mean);
    bandpass_std = sqrt(bandpass_std / survey -> nchans);

    // Calculate root mean square error between fitted and original bandpass
    survey -> bandpass_rmse = 0;
    for(unsigned i = 0; i < survey -> nchans; i++)
        survey -> bandpass_rmse += pow(bandpass[i] - fitted_bandpass[i], 2);

    survey -> bandpass_rmse = sqrt(survey -> bandpass_rmse / survey -> nchans);
    survey -> bandpass_mean = bandpass_mean;
    survey -> bandpass_std = bandpass_std;

    // If required, show bandpass plot
    #if SHOW_BANDPASS
    if (params -> thread_num == 0)
    {
        unsigned nchans = survey -> nchans;
        float x_vals[nchans], y_vals[nchans], orig_vals[nchans];
        float y_min = 10e9, y_max = -10e9;
        for(unsigned i = 0; i < nchans; i++)
        {
            x_vals[i] = 0 + i * (1.0 / nchans);
            y_vals[i] = (float) bandpass[i];
            orig_vals[i] = (float) fitted_bandpass[i];
            y_min = (y_vals[i] < y_min) ? y_vals[i] : y_min;
            y_min = (orig_vals[i] < y_min) ? orig_vals[i] : y_min;
            y_max = (y_vals[i] > y_max) ? y_vals[i] : y_max;
            y_max = (orig_vals[i] > y_max) ? orig_vals[i] : y_max;
        }
    
        if (y_min == y_max) y_max += 0.00001;

        cpgenv(0, 1, y_min, y_max, 0, 1);
        cpgsci(7);
        cpgline(nchans, x_vals, y_vals);
        cpgsci(8);
        cpgline(nchans, x_vals, orig_vals);
        cpgmtxt("T", 2.0, 0.0, 0.0, "Bandpass Plot");
    }
    #endif

    // Wait for memcpy to finish
    cudaThreadSynchronize();

    // And we're done
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("%d: Calculated bandpass in [Beam %d]: %lf\n", (int) (time(NULL) - params -> start), 
                                                          params -> thread_num, timestamp);
}

// Calculate bandpass and store relevant information on CPU/GPU
void rfi_clipping(float *d_input, double *d_bandpass, THREAD_PARAMS* params, cudaEvent_t event_start, 
                  cudaEvent_t event_stop, unsigned shift, unsigned nsamp, unsigned total)
{
    SURVEY *survey = params -> survey;
    float timestamp, tot_timestamp = 0;    
    
    // Calculate rejection thresholds
    float channel_thresh  = survey -> channel_thresh * survey -> bandpass_rmse;
    float spectrum_thresh = survey -> bandpass_mean + survey -> spectrum_thresh * survey -> bandpass_std;

    // Allocate temporary GPU buffer for channel flags
    char *d_flags;
    unsigned num_blocks = ceil(nsamp / (float) survey -> channel_block);

    cudaEventRecord(event_start, 0);
    cudaMalloc((void **) &d_flags, survey -> nchans * num_blocks * sizeof(char));
    cudaMemset(d_flags, 0, survey -> nchans * num_blocks * sizeof(char));
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
    tot_timestamp += timestamp;

    // Flag channel blocks
    cudaEventRecord(event_start, 0);
    channel_flagger<<< dim3(num_blocks, survey -> nchans), BANDPASS_THREADS >>>
                   (d_input, d_bandpass, d_flags, nsamp, survey -> nchans, survey -> channel_block, num_blocks, channel_thresh, total, shift);
    cudaMemset(d_flags, 0, survey -> nchans * num_blocks * sizeof(char));
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
    tot_timestamp += timestamp;

    // Clip channel blocks
    cudaEventRecord(event_start, 0);
    channel_clipper<<< dim3(num_blocks, survey -> nchans), BANDPASS_THREADS >>>
                   (d_input, d_bandpass, d_flags, nsamp, survey -> nchans, survey -> channel_block, num_blocks, total, shift);
    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
    tot_timestamp += timestamp;

    // Free GPU channel flags
    cudaFree(d_flags);

    // Clip spectra
    cudaEventRecord(event_start, 0);
    spectrum_clipper<<< nsamp / BANDPASS_THREADS, BANDPASS_THREADS >>>
                   (d_input, d_bandpass, survey -> bandpass_mean, nsamp, survey -> nchans, 
                    shift, total, spectrum_thresh, survey -> bandpass_mean - survey -> spectrum_thresh * survey -> bandpass_std);

    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
    tot_timestamp += timestamp;
   
    // All done

	printf("%d: Clipped RFI [Beam %d]: %lf\n", (int) (time(NULL) - params -> start), params -> thread_num, tot_timestamp);
}


// Perform median-filtering on dedispersed-time series
void apply_median_filter(float *d_input, THREAD_PARAMS* params, cudaEvent_t event_start, cudaEvent_t event_stop, unsigned nsamp)
{
    SURVEY *survey = params -> survey;
    unsigned tid = params -> thread_num;
    float timestamp;

    cudaEventRecord(event_start, 0);	

    // Apply median filter on GPU
    dim3(nsamp / MEDIAN_THREADS, survey -> tdms); 
    median_filter<<<dim3(nsamp / MEDIAN_THREADS, survey -> tdms), MEDIAN_THREADS>>>
                   (d_input, nsamp);

    // All processing ready, wait for kernel execution
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("%d: Performed Median-Filtering [Beam %d]: %lf\n", (int) (time(NULL) - params -> start), 
                                                              tid, timestamp);
}

// Detrend dedispersion time series
void apply_detrending(float *d_input, THREAD_PARAMS* params, cudaEvent_t event_start, cudaEvent_t event_stop, unsigned nsamp)
{
    SURVEY *survey = params -> survey;
    unsigned tid = params -> thread_num;
    float timestamp;

    cudaEventRecord(event_start, 0);	
    unsigned detrend = nsamp;
	detrend_normalise<<<dim3(ceil(nsamp / (1.0 * detrend)), survey -> tdms), BANDPASS_THREADS>>>(d_input, detrend);

    // All processing ready, wait for kernel execution
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("%d: Performed Detrending [Beam %d]: %lf\n", (int) (time(NULL) - params -> start), 
                                                              tid, timestamp);
}

// Detrend dedispersion time series
void perform_beamforming(float *d_input, float *d_output, float *d_shifts, THREAD_PARAMS* params, 
                         cudaEvent_t event_start, cudaEvent_t event_stop, unsigned nsamp, unsigned shift)
{
    SURVEY *survey = params -> survey;
    unsigned tid = params -> thread_num;
    float timestamp;

    cudaEventRecord(event_start, 0);	

    beamform_medicina<<< dim3(nsamp / BEAMFORMER_THREADS, survey -> nchans, BEAMS / BEAMS_PER_TB), BEAMFORMER_THREADS >>>
					     ((char4 *) d_input, d_output, d_shifts, nsamp, survey -> nchans, shift);

    // All processing ready, wait for kernel execution
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    CudaSafeCall(cudaThreadSynchronize());
    printf("%d: Performed beamforming [Beam %d]: %lf\n", (int) (time(NULL) - params -> start), 
                                                              tid, timestamp);
}


// =================================== CUDA CPU THREAD MAIN FUNCTION ====================================
void* dedisperse(void* thread_params)
{
    THREAD_PARAMS* params = (THREAD_PARAMS *) thread_params;
    SURVEY *survey = params -> survey;
    BEAM beam = (params -> survey -> beams)[params -> thread_num];
    GPU *gpu = (params -> gpus)[params -> gpu_index];
    int i, nsamp = params -> survey -> nsamp, nchans = params -> survey -> nchans, nants = survey -> nantennas;
    int loop_counter = 0, maxshift = beam.maxshift, iters = 0, tid = params -> thread_num;
    time_t start = params -> start;

    printf("%d: Started thread %d [GPU %d]\n", (int) (time(NULL) - start), tid, gpu -> device_id);

    // Initialise device
    CudaSafeCall(cudaSetDevice(gpu -> device_id));
    CudaSafeCall(cudaDeviceReset());
    cudaSetDeviceFlags(cudaDeviceBlockingSync);

    // Wait at GPU barrier for primary thread to finish allocating GPU memory
    int ret = pthread_barrier_wait(&(gpu -> barrier));
    if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
        { fprintf(stderr, "Error during primary thread synchronisation\n"); exit(0); }

    // Avoid initialisation conflicts
    sleep(tid);

    // Only the master thread per GPU can allocate input and output buffer (which is useful for 
    // performing cross-GPU processing such as beamforming
    float *beamshifts, *d_beamshifts;
    if (gpu -> primary_thread == tid)
    {
        // Allocate output buffer
        float *d_output;
        if (survey -> apply_beamforming)
            CudaSafeCall(cudaMalloc((void **) &d_output, (unsigned long) nants * nchans * nsamp * sizeof(unsigned char)));
        else
            CudaSafeCall(cudaMalloc((void **) &d_output, survey -> tdms * nsamp * sizeof(float) * gpu -> num_threads));

        // Allocate input buffer
        float *d_input;
        CudaSafeCall(cudaMalloc((void **) &d_input, params -> inputsize * gpu -> num_threads));

        // Allocate shift buffers (only GPU's primary thread can use this memory buffer)
        CudaSafeCall(cudaMallocHost((void **) &beamshifts, nants * nchans * gpu -> num_threads * sizeof(float), cudaHostAllocPortable));
        CudaSafeCall(cudaMalloc((void **) &d_beamshifts, nants * nchans * gpu -> num_threads * sizeof(float)));
        CudaSafeCall(cudaMemset(d_beamshifts, 0, nants * nchans * gpu -> num_threads * sizeof(float))); 

        // Distribute pointers to all threads assigned to the same GPU
        for(unsigned i = 0; i < gpu -> num_threads; i++)
        {
            // inputsize and outputsize are in bytes, while pointer arithmetic works with
            // words, so we must divide with the word size
            THREAD_PARAMS *curr_thread = (params -> cpu_threads)[(gpu -> thread_ids)[i]];
            curr_thread -> d_input = d_input + params -> inputsize * i / sizeof(void *);    
            curr_thread -> d_output = d_output + (survey -> tdms * survey -> nsamp * sizeof(float)) * i / sizeof(void *);             
        }
    }

    // Wait at GPU barrier for primary thread to finish allocating GPU memory
    ret = pthread_barrier_wait(&(gpu -> barrier));
    if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
        { fprintf(stderr, "Error during primary thread synchronisation\n"); exit(0); }

    // All thread can then allocate their beam-specific temporary buffer (shifts, bandpass etc...)

    // Allocate output buffer to store dedispersed time series
    {
        float *output_buffer;
        CudaSafeCall(cudaHostAlloc(&output_buffer, survey -> nsamp * survey -> tdms * sizeof(float), cudaHostAllocPortable));
        params -> output[params -> thread_num] = output_buffer;
    }

    // Allocate device memory and copy dmshifts and dmvalues to constant memory
    float *d_input = params -> d_input, *d_output = params -> d_output, *d_dmshifts;
    CudaSafeCall(cudaMalloc((void **) &d_dmshifts, nchans * sizeof(float)));
    CudaSafeCall(cudaMemcpy(d_dmshifts, beam.dm_shifts, nchans * sizeof(float), cudaMemcpyHostToDevice));

    // Cacluate the extra shared memory required to store shifts
    unsigned shared_shift = round(beam.dm_shifts[survey -> nchans - 1] * (survey -> lowdm + survey -> tdms * survey -> dmstep) / survey -> tsamp) -     
                            round(beam.dm_shifts[survey -> nchans - 1] * (survey -> lowdm + (survey -> tdms - DEDISP_DMS) * survey -> dmstep) / survey -> tsamp);
    int *d_all_shifts;

    if (1)
    {
	    // Pre-compute channel and DM specific shifts beforehand on CPU
	    // This only needs to be computed once for the entire execution
	    int *all_shifts = (int *) malloc(survey -> nchans * survey -> tdms * sizeof(int));
	    for(unsigned c = 0; c < survey -> nchans; c++)
		    for (unsigned d = 0; d < survey -> tdms; d++)
		        all_shifts[c * survey -> tdms + d] = (int) (beam.dm_shifts[c] * (d * survey -> dmstep) / survey -> tsamp);

	    
	    CudaSafeCall(cudaMalloc((void **) &d_all_shifts, survey -> nchans * survey -> tdms * sizeof(int)));
	    CudaSafeCall(cudaMemcpy(d_all_shifts, all_shifts, survey -> nchans * survey -> tdms * sizeof(int), cudaMemcpyHostToDevice) ); 
        free(all_shifts);
    }

   
    // Bandpass-related CPU and GPU buffers
    double *bandpass, *d_bandpass;
    CudaSafeCall(cudaMallocHost((void **) &bandpass, nchans * sizeof(double), cudaHostAllocPortable));
    CudaSafeCall(cudaMalloc((void **) &d_bandpass, nchans * sizeof(double)));
    CudaSafeCall(cudaMemset(d_bandpass, 0, nchans * sizeof(double)));    

    // Set CUDA kernel preferences
    CudaSafeCall(cudaFuncSetCacheConfig(cache_dedispersion, cudaFuncCachePreferL1 ));
    CudaSafeCall(cudaFuncSetCacheConfig(median_filter, cudaFuncCachePreferShared ));

    // Initialise events / performance timers
    cudaEvent_t event_start, event_stop;
    float timestamp;

    cudaEventCreate(&event_start);
    
    // Blocking sync when waiting for kernel launches
    cudaEventCreateWithFlags(&event_stop, cudaEventBlockingSync); 

    // Initialise PG Plotter if required
    #if SHOW_BANDPASS
        if (params->survey->apply_rfi_clipper && tid == 0)
        {
            if(cpgbeg(0, "/xwin", 1, 1) != 1)
                printf("Couldn't initialise PGPLOT\n");
            cpgask(false);
        }
    #endif

    // Store pointer for current buffer
    float *input_ptr;

    // Thread processing loop
    while (1)
    {
        //  Read input data into GPU memory ===================================
        if (loop_counter >= params -> iterations - 1) 
        {
            // Update global input pointer
            input_ptr = (params -> input)[(loop_counter - 1) % MDSM_STAGES] + 
                        nchans * nsamp * beam.beam_id;

            // Start recording CPU-GPU IO time
            cudaEventRecord(event_start, 0);

            // First iteration
            if (loop_counter == 1)
            {
                // If beamforming, copy input data antenna data to output buffer on GPU
                if (survey -> apply_beamforming)
                {
                    if (gpu -> primary_thread == tid) // Only primary thread per GPU performs processing
                    {
                        // We need to beamform maxshift spectra and place them at the end of each respective
                        // beam/channel. These will be copied to the end front of the buffer in the next iteration
                        unsigned char *char_ptr = (unsigned char *) d_output;
                        for(i = 0; i < nchans; i++)
                            CudaSafeCall(cudaMemcpyAsync(char_ptr + nants * ((nsamp + maxshift) * i + nsamp),
                                                               params -> antenna_buffer + nsamp * nants * i,
                                                               maxshift * nants * sizeof(unsigned char),
                                                               cudaMemcpyHostToDevice));

                        CudaSafeCall(cudaThreadSynchronize()); // Wait for all copies
                    }

                    // Wait at GPU barrier for primary thread to finish copying input antenna data
                    ret = pthread_barrier_wait(&(gpu -> barrier));
                    if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
                        { fprintf(stderr, "Error during primary thread synchronisation\n"); exit(0); }
                }
                else
                {
                    // If not performing beamforming, just copy maxshift spectra at the end of each channel (they
                    // will be copied to the front of the buffer during the next iteration
                    for (i = 0; i < nchans; i++)
                        CudaSafeCall(cudaMemcpyAsync(d_input + (nsamp + maxshift) * i + nsamp, 
                                                input_ptr + nsamp * i + (nsamp - maxshift), 
                                                maxshift * sizeof(float), cudaMemcpyHostToDevice));

                    CudaSafeCall(cudaThreadSynchronize());  // Wait for all copies
                }
            }
            else  // Not the first iteration
            {
                // Copy maxshift to beginning of buffer (in each channel)
                for(i = 0; i < nchans; i++)
                    CudaSafeCall(cudaMemcpyAsync(d_input + (nsamp + maxshift) * i, 
                                                 d_input + (nsamp + maxshift) * i + nsamp, 
                                                 maxshift * sizeof(float), cudaMemcpyDeviceToDevice));

                // If beamforming copy the entire antenna data to the output buffer
                if (survey -> apply_beamforming)
                {
                    if (gpu -> primary_thread == tid) // Only primary thread per GPU performs processing
                        CudaSafeCall(cudaMemcpyAsync(d_output, params -> antenna_buffer, (unsigned long) nsamp * nchans * nants * sizeof(unsigned char), 
                                                cudaMemcpyHostToDevice));

                    // Wait at GPU barrier for primary thread to finish copying input antenna data
                    ret = pthread_barrier_wait(&(gpu -> barrier));
                    if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
                        { fprintf(stderr, "Error during primary thread synchronisation\n"); exit(0); }
                }
                else
                {
                    // Wait for maxshift copying to avoid data inconsistencies
                    CudaSafeCall(cudaThreadSynchronize());

                    // Copy nsamp from each channel to GPU (ignoring first maxshift samples)
                    for(i = 0; i < nchans; i++)
                        CudaSafeCall(cudaMemcpyAsync(d_input + (nsamp + maxshift) * i + maxshift, 
                                                     input_ptr + nsamp * i,
                                                     nsamp * sizeof(float), cudaMemcpyHostToDevice));
                }
            }

            cudaEventRecord(event_stop, 0);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&timestamp, event_start, event_stop);
            printf("%d: Copied data to GPU [Beam %d]: %f\n", (int) (time(NULL) - start), tid, timestamp);
        }

        // Wait input barrier
        int ret = pthread_barrier_wait(params -> input_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
            { fprintf(stderr, "Error during barrier synchronisation 1 [thread]\n"); exit(0); }

        //  Perform computation on GPUs: 1st Iteration ===================================
        if (loop_counter == 1)
        {
                // First beamforming iteration, so we need to beamform maxshift samples to their respective
                // position in the input buffer
                if (survey -> apply_beamforming)
                {
                    if (gpu -> primary_thread == tid) // Only primary thread per GPU performs processing
                        perform_beamforming(d_output, d_input, d_beamshifts, params, event_start, event_stop, maxshift, nsamp);

                    // Wait at GPU barrier for primary thread to finish allocating GPU memory
                    ret = pthread_barrier_wait(&(gpu -> barrier));
                    if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
                        { fprintf(stderr, "Error during primary thread synchronisation\n"); exit(0); }
                }
                else
                {
                    // This is the first iteration, so if we have complex voltages we need to calcualte
                    // their power for the first maxshift input spectra
                    if (params -> survey -> voltage)   
                        calculate_power(d_input, params, event_start, event_stop, nsamp, 
                                        maxshift, nsamp + maxshift);
                }

                // This is the first iteration, so if Bandpass Fitting and RFI clipping is
                // required we'll need to do it here
                if (params -> survey -> apply_rfi_clipper)
                {
                    bandpass_fitting(d_input, bandpass, d_bandpass, params, event_start, 
                                     event_stop, nsamp, maxshift, nsamp + maxshift);
                    rfi_clipping(d_input, d_bandpass, params, event_start, event_stop, 
                                 nsamp, maxshift, nsamp + maxshift);
                }
        }

        //  Perform computation on GPUs: Rest ===========================================
        else if (loop_counter >= params -> iterations)
        {
            // If input data is complex voltage, we need to compute the power for each sample
            if (params -> survey -> voltage)   
                calculate_power(d_input, params, event_start, event_stop, 
                                maxshift, nsamp, nsamp + maxshift);

            if (survey -> apply_beamforming)
            {
                if (gpu -> primary_thread == tid) // Only primary thread per GPU performs processing
                    perform_beamforming(d_output, d_input, d_beamshifts, params, event_start, event_stop, nsamp, maxshift);

                // Wait at GPU barrier for primary thread to finish allocating GPU memory
                ret = pthread_barrier_wait(&(gpu -> barrier));
                if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
                    { fprintf(stderr, "Error during primary thread synchronisation\n"); exit(0); }
            }

            if (params -> survey -> apply_rfi_clipper)
            {
                // Calculate Bandpass
                bandpass_fitting(d_input, bandpass, d_bandpass, params, event_start, 
                                 event_stop, maxshift, nsamp, nsamp + maxshift);

                // Perform RFI clipping
                rfi_clipping(d_input, d_bandpass, params, event_start, event_stop, 
                             maxshift, nsamp, nsamp + maxshift);
            }

            // Wait at GPU barrier for primary thread to finish allocating GPU memory
            int ret = pthread_barrier_wait(&(gpu -> barrier));
            if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
                { fprintf(stderr, "Error during primary thread synchronisation\n"); exit(0); }

            // Perform Dedispersion
            if (0)
    		    cached_brute_force(d_input, d_output, d_dmshifts, params, 
                                   event_start, event_stop, nsamp, beam.maxshift);
            else
    		    shared_brute_force(d_input, d_output, d_all_shifts, params, 
                                   event_start, event_stop, nsamp, beam.maxshift, shared_shift);

            // Apply median filter if required
            if (params -> survey -> apply_median_filter)
                apply_median_filter(d_output, params, event_start, event_stop, nsamp);

            // Apply detrending and normalisation
            if (params -> survey -> apply_detrending)
                apply_detrending(d_output, params, event_start, event_stop, nsamp);
        }

        // Wait output barrier
        ret = pthread_barrier_wait(params -> output_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
            { fprintf(stderr, "Error during barrier synchronisation 2 [thread]\n"); exit(0); }

        //  Output results back to CPU =====================================
        if (loop_counter >= params -> iterations ) 
        { 
            // Collect and write output to host memory
            // TODO: Overlap this copy with the input part of this thread
            cudaEventRecord(event_start, 0);
            CudaSafeCall(cudaMemcpy( params -> output[params -> thread_num], d_output, 
            						 survey -> tdms * survey -> nsamp * sizeof(float),
                                     cudaMemcpyDefault));
            cudaEventRecord(event_stop, 0);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&timestamp, event_start, event_stop);
            printf("%d: Copied data from GPU [Beam %d]: %f\n", 
                   (int) (time(NULL) - start), tid, timestamp);
        }

        // Acquire rw lock
        if (pthread_rwlock_rdlock(params -> rw_lock))
            { fprintf(stderr, "Unable to acquire rw_lock [thread]\n"); exit(0); }

        // Update params  
        nsamp = params -> survey -> nsamp;

        // Stopping clause
        if (((THREAD_PARAMS *) thread_params) -> stop) 
        {
            if (iters >= params -> iterations - 1)
            {
                // Release rw_lock
                if (pthread_rwlock_unlock(params -> rw_lock))
                    { fprintf(stderr, "Error releasing rw_lock [thread]\n"); exit(0); }

                for(i = 0; i < params -> maxiters - params -> iterations ; i++) 
                {
                    pthread_barrier_wait(params -> input_barrier);
                    pthread_barrier_wait(params -> output_barrier);
                }

                break; 
            }
            else
                iters++;
        }

        // Release rw_lock
        if (pthread_rwlock_unlock(params -> rw_lock))
            { fprintf(stderr, "Error releasing rw_lock [thread]\n"); exit(0); }

        loop_counter++;
    }   

    CudaSafeCall(cudaFree(d_output));
    CudaSafeCall(cudaFree(d_input));
    cudaEventDestroy(event_stop);
    cudaEventDestroy(event_start); 

    printf("%d: Exited gracefully %d\n", (int) (time(NULL) - start), tid);
    pthread_exit((void*) thread_params);
}
