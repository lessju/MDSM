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
{  CudaSafeCall(cudaMallocHost((void **) pointer, size, cudaHostAllocPortable));  }

// Allocate memory-pinned output buffer
void allocateOutputBuffer(float **pointer, size_t size)
{ CudaSafeCall(cudaMallocHost((void **) pointer, size, cudaHostAllocPortable)); }


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

    printf("Bandpass fit chisq: %lf\n", chisq);
 
    /* store result ... */
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
                        cudaEvent_t event_start, cudaEvent_t event_stop, int maxshift)
{
    SURVEY *survey = params -> survey;

    int num_reg         = NUMREG;
    int divisions_in_t  = DIVINT;
    int divisions_in_dm = DIVINDM;
    int num_blocks_t    = (survey -> nsamp / (divisions_in_t * num_reg));
    int num_blocks_dm   = survey -> tdms / divisions_in_dm;

    float timestamp;       
    dim3 threads_per_block(divisions_in_t, divisions_in_dm);
    dim3 num_blocks(num_blocks_t,num_blocks_dm); 

    cudaEventRecord(event_start, 0);	

    cache_dedispersion<<< num_blocks, threads_per_block >>>
                      (d_output, d_input, d_dmshifts, survey -> nsamp, 
                       survey -> nchans, survey -> lowdm / survey -> tsamp, 
                       survey -> dmstep/survey -> tsamp, maxshift);

    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("%d: Performed Brute-Force Dedispersion [Beam %d]: %lf\n", 
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

    // Fit polynomial using GNU Scientific Library
    polynomialfit(survey -> nchans, survey -> ncoeffs, X, bandpass, coeffs); 

    // Generate 1D polynomial using bandpass co-efficients
    // We also need the fit-corrected bandpass to compute the bandpass RMS
    double corrected_bandpass[survey -> nchans], summed_bandpass[survey -> nchans];
    
    // Copy bandpass to corrected_bandpass
    memcpy(corrected_bandpass, bandpass, survey -> nchans * sizeof(double)); 
    memcpy(summed_bandpass, bandpass, survey -> nchans * sizeof(double)); 
    memset(bandpass, 0, survey -> nchans * sizeof(double));

    for(unsigned i = 0; i < survey -> nchans; i++)
    {
        for(unsigned j = 0; j < survey -> ncoeffs; j++)
            bandpass[i] += coeffs[j] * pow(X[i], j);
        corrected_bandpass[i] -= bandpass[i];
    }

    // Asynchronous copy of bandpass
    CudaSafeCall(cudaMemcpyAsync(d_bandpass, bandpass, survey -> nchans * sizeof(double), 
                 cudaMemcpyHostToDevice));

    // Calculate bandpass statistics to be used later on    
    float corr_bandpass_mean = 0, corr_bandpass_std = 0, corr_bandpass_rms = 0;
    float bandpass_mean = 0, bandpass_std = 0, bandpass_rms = 0;

    // First iteration to compute mean
    for(unsigned i = 0; i < survey -> nchans; i++)
    {
        bandpass_mean += bandpass[i];
        bandpass_rms += bandpass[i] * bandpass[i];
        corr_bandpass_mean += corrected_bandpass[i];
        corr_bandpass_rms += corrected_bandpass[i] * corrected_bandpass[i];
    }
    bandpass_mean /= survey -> nchans;
    bandpass_rms = sqrt(bandpass_rms / survey -> nchans);
    corr_bandpass_mean /= survey -> nchans;
    corr_bandpass_rms = sqrt(corr_bandpass_rms / survey -> nchans);
    
    // Second iteration, compute standard deviation
    for(unsigned i = 0; i < survey -> nchans; i++)
    {
        bandpass_std += (bandpass[i] - bandpass_mean) * (bandpass[i] - bandpass_mean);
        corr_bandpass_std += (bandpass[i] - corr_bandpass_mean) * (bandpass[i] - corr_bandpass_mean);
    }
    bandpass_std = sqrt(bandpass_std / survey -> nchans);
    corr_bandpass_std = sqrt(corr_bandpass_std / survey -> nchans);

    survey -> corrected_bandpass_mean = corr_bandpass_mean;
    survey -> corrected_bandpass_std = corr_bandpass_std;
    survey -> corrected_bandpass_rms = corr_bandpass_rms;
    survey -> bandpass_mean = bandpass_mean;
    survey -> bandpass_std = bandpass_std;
    survey -> bandpass_rms = bandpass_rms;

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
            orig_vals[i] = (float) summed_bandpass[i];
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

//    printf("Bandpass. Mean: %f, Std: %f, RMS: %f\n", bandpass_mean, bandpass_std, bandpass_rms);
//    printf("Corrected Bandpass. Mean: %f, Std: %f, RMS: %f\n", corr_bandpass_mean,
//                                    corr_bandpass_std, corr_bandpass_rms);

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
    float timestamp;    
    
    // Calculate rejection thresholds
    float channel_thresh  = survey -> channel_thresh * survey -> corrected_bandpass_rms;
    float spectrum_thresh = survey -> spectrum_thresh *  survey -> corrected_bandpass_rms;

//    printf("Channel thresh: %f, spectrum thresh: %f\n", channel_thresh, spectrum_thresh);

    cudaEventRecord(event_start, 0);
    channel_clipper<<< dim3(ceil(nsamp / (float) survey -> channel_block), survey -> nchans), BANDPASS_THREADS >>>
                   (d_input, d_bandpass, survey -> bandpass_mean, survey -> channel_block, nsamp, 
                    survey -> nchans, shift, total, channel_thresh);

    cudaThreadSynchronize();

    spectrum_clipper<<< nsamp / BANDPASS_THREADS, BANDPASS_THREADS >>>
                   (d_input, d_bandpass, survey -> bandpass_mean, nsamp, survey -> nchans, 
                    shift, total, spectrum_thresh);

    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("%d: Clipped RFI [Beam %d]: %lf\n", (int) (time(NULL) - params -> start), params -> thread_num, timestamp);
}


// Perform median-filtering on dedispersed-time series
void apply_median_filter(float *d_input, THREAD_PARAMS* params, 
                         cudaEvent_t event_start, cudaEvent_t event_stop)
{
    SURVEY *survey = params -> survey;
    unsigned tid = params -> thread_num;
    float timestamp;

    cudaEventRecord(event_start, 0);	

    // Apply median filter on GPU
    dim3(survey -> nsamp / MEDIAN_THREADS, survey -> tdms); 
    median_filter<<<dim3(survey -> nsamp / MEDIAN_THREADS, survey -> tdms), MEDIAN_THREADS>>>
                   (d_input, survey -> nsamp);

    // All processing ready, wait for kernel execution
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("%d: Performed Median-Filtering [Beam %d]: %lf\n", (int) (time(NULL) - params -> start), 
                                                              tid, timestamp);
}

// Detrend dedispersion time series
void apply_detrending(float *d_input, THREAD_PARAMS* params, 
                         cudaEvent_t event_start, cudaEvent_t event_stop)
{
    SURVEY *survey = params -> survey;
    unsigned tid = params -> thread_num;
    float timestamp;

    cudaEventRecord(event_start, 0);	
    unsigned detrend = survey -> nsamp;
	detrend_normalise<<<dim3(ceil(survey -> nsamp / (1.0 * detrend)), survey -> tdms), BANDPASS_THREADS>>>(d_input, detrend);

    // All processing ready, wait for kernel execution
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("%d: Performed Detrending [Beam %d]: %lf\n", (int) (time(NULL) - params -> start), 
                                                              tid, timestamp);
}

// =================================== CUDA CPU THREAD MAIN FUNCTION ====================================
void* dedisperse(void* thread_params)
{
    THREAD_PARAMS* params = (THREAD_PARAMS *) thread_params;
    BEAM beam = (params -> survey -> beams)[params -> thread_num];
    int i, nsamp = params -> survey -> nsamp, nchans = params -> survey -> nchans;
    int loop_counter = 0, maxshift = beam.maxshift, iters = 0, tid = params -> thread_num;
    time_t start = params -> start;

    printf("%d: Started thread %d [GPU %d]\n", (int) (time(NULL) - start), tid, beam.gpu_id);

    // Initialise device
    CudaSafeCall(cudaSetDevice(beam.gpu_id));
    CudaSafeCall(cudaDeviceReset());
    cudaSetDeviceFlags( cudaDeviceBlockingSync );

    // Avoid initialisation conflicts
    sleep(1);

    // Allocate device memory and copy dmshifts and dmvalues to constant memory
    float *d_input, *d_output, *d_dmshifts;
    CudaSafeCall(cudaMalloc((void **) &d_input, params -> inputsize));
    CudaSafeCall(cudaMalloc((void **) &d_output, params -> outputsize));
    CudaSafeCall(cudaMalloc((void **) &d_dmshifts, nchans * sizeof(float)));
    CudaSafeCall(cudaMemcpy(d_dmshifts, beam.dm_shifts, nchans * sizeof(float), cudaMemcpyHostToDevice));
   
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
            // Update input pointer
            input_ptr = (params -> input)[(loop_counter - 1) % MDSM_STAGES] + 
                        nchans * nsamp * beam.beam_id;

            cudaEventRecord(event_start, 0);
            if (loop_counter == 1)
            {
                // First iteration, just copy maxshift spectra at the end of each channel (they
                // will be copied to the front of the buffer during the next iteration)
                for (i = 0; i < nchans; i++)
                    CudaSafeCall(cudaMemcpyAsync(d_input + (nsamp + maxshift) * i + nsamp, 
                                                 input_ptr + nsamp * i + (nsamp - maxshift), 
                                                 maxshift * sizeof(float), cudaMemcpyHostToDevice));

                CudaSafeCall(cudaThreadSynchronize());  // Wait for all copies
            }
            else 
            {
                // Copy maxshift to beginning of buffer (in each channel)
                for(i = 0; i < nchans; i++)
                    CudaSafeCall(cudaMemcpyAsync(d_input + (nsamp + maxshift) * i, 
                                                 d_input + (nsamp + maxshift) * i + nsamp, 
                                                 maxshift * sizeof(float), cudaMemcpyDeviceToDevice));

                // Wait for maxshift copying to avoid data inconsistencies
                CudaSafeCall(cudaThreadSynchronize());

                // Copy nsamp from each channel to GPU (ignoring first maxshift samples)
                for(i = 0; i < nchans; i++)
                    CudaSafeCall(cudaMemcpyAsync(d_input + (nsamp + maxshift) * i + maxshift, 
                                                 input_ptr + nsamp * i,
                                                 nsamp * sizeof(float), cudaMemcpyHostToDevice));
            }

            cudaEventRecord(event_stop, 0);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&timestamp, event_start, event_stop);
            printf("%d: Copied data to GPU [Beam %d]: %f\n", (int) (time(NULL) - start), tid, timestamp);

            // Clear GPU output buffer
            CudaSafeCall(cudaMemset(d_output, 0, params -> outputsize));
        }

        // Wait input barrier
        int ret = pthread_barrier_wait(params -> input_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
            { fprintf(stderr, "Error during barrier synchronisation 1 [thread]\n"); exit(0); }

        //  Perform computation on GPUs: 1st Iteration ===================================
        if (loop_counter == 1)
        {
                // This is the first iteration, so if we have complex voltages we need to calcualte
                // their power for the first maxshift input spectra
                if (params -> survey -> voltage)   
                    calculate_power(d_input, params, event_start, event_stop, nsamp, 
                                    maxshift, nsamp + maxshift);

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

            if (params -> survey -> apply_rfi_clipper)
            {
                // Calculate Bandpass
                bandpass_fitting(d_input, bandpass, d_bandpass, params, event_start, 
                                 event_stop, maxshift, nsamp, nsamp + maxshift);

                // Perform RFI clipping
                rfi_clipping(d_input, d_bandpass, params, event_start, event_stop, 
                             maxshift, nsamp, nsamp + maxshift);
            }

//            if (loop_counter >= params -> iterations)
//            {
//                // =========== TEMP: Stop here for testing
//                float *temp = (float *) malloc(nchans * (nsamp+maxshift) * sizeof(float));
//                for(unsigned i =0; i < nchans; i++)
//                      CudaSafeCall(cudaMemcpy( &temp[i * (nsamp+maxshift)], 
//                                               &d_input[i * (nsamp + maxshift)], 
//                                               (nsamp+maxshift) * sizeof(float),      
//                                               cudaMemcpyDeviceToHost));
//                char filename[256];
//                char beam_no[2];
//                sprintf(beam_no, "%d", beam.beam_id);
//                strcat(filename, "Test_RFI_");
//                strcat(filename, beam_no);
//                strcat(filename, ".dat");
//                FILE *fp = fopen(filename, "wb");
//                fwrite(temp, sizeof(float), nchans * (nsamp+maxshift), fp);
//                free(temp);
//                fclose(fp);
//                sleep(3);
//                exit(0);            
//            }

            // Perform Dedispersion
		    cached_brute_force(d_input, d_output, d_dmshifts, params, 
                                event_start, event_stop, beam.maxshift);

            // Apply median filter if required
            if (params -> survey -> apply_median_filter)
                apply_median_filter(d_output, params, event_start, event_stop);

            // Apply detrending and normalisation
            if (params -> survey -> apply_detrending)
                apply_detrending(d_output, params, event_start, event_stop);
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
            CudaSafeCall(cudaMemcpy( params -> output, d_output, 
            						 params -> dedispersed_size * sizeof(float),
                                     cudaMemcpyDeviceToHost));
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
