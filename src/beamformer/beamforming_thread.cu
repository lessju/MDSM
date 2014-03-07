#include "beamforming_kernel.cu"
#include "beamforming_thread.h"
#include "math.h"
#include "cufft.h"

// ===================== CUDA HELPER FUNCTIONS ==========================

// Error checking function
#define CUDA_ERROR_CHECK
#define CudaSafeCall( err ) _cudaSafeCall( err, __FILE__, __LINE__ )
#define CufftSafeCall( err ) _cufftSafeCall( err, __FILE__, __LINE__ )
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


inline void _cufftSafeCall(cufftResult err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
    if( CUFFT_SUCCESS != err) 
    {
        fprintf(stderr, "cufftCheckError() failed at %s:%i : %d\n", file, line , err); 
        cudaDeviceReset(); exit(-1);
    }
#endif
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

// =================================== CUDA KERNEL HELPERS ====================================

// Wrapper for beamforming kernel
void perform_beamforming(unsigned char *d_input, float *d_output, float2 *d_shifts, THREAD_PARAMS* params, 
                         cudaEvent_t event_start, cudaEvent_t event_stop, unsigned nsamp)
{
    SURVEY *survey = params -> survey;
    unsigned tid = params -> thread_num;
    float timestamp;

    cudaEventRecord(event_start, 0);	

    if (survey -> perform_channelisation)
        beamformer_complex<<< dim3(nsamp / BEAMFORMER_THREADS, survey -> nchans, BEAMS / BEAMS_PER_TB), BEAMFORMER_THREADS >>>
					              ((char4 *) d_input, (float2 *) d_output, d_shifts, nsamp, survey -> nchans);
    else
        beamformer<<< dim3(nsamp / BEAMFORMER_THREADS, survey -> nchans, BEAMS / BEAMS_PER_TB), BEAMFORMER_THREADS >>>
					         ((char4 *) d_input, d_output, d_shifts, nsamp, survey -> nchans);

    // All processing ready, wait for kernel execution
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    CudaSafeCall(cudaThreadSynchronize());
    printf("%d: Performed Beamforming [Thread %d]: %lf\n", (int) (time(NULL) - params -> start), 
                                                              tid, timestamp);

}

// Wrapper for data rearrangement kernel
void perform_rearrangement(unsigned char *input, unsigned char *output, THREAD_PARAMS *params, cudaEvent_t event_start, 
                           cudaEvent_t event_stop, unsigned nsamp)
{
    SURVEY *survey = params -> survey;
    unsigned tid = params -> thread_num;
    float timestamp;

    cudaEventRecord(event_start, 0);	

	dim3 gridDim(nsamp / HEAP, survey -> nchans);  
    rearrange_medicina<<< gridDim, HEAP >>> (input, output, nsamp, survey -> nchans);

    // All processing ready, wait for kernel execution
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    CudaCheckError();
    printf("%d: Performed Re-Arrangement [Thread %d]: %lf\n", (int) (time(NULL) - params -> start), 
                                                              tid, timestamp);
}

// Wrapper for integration kernel
void perform_downsampling(float *input, float *output, THREAD_PARAMS* params, cudaEvent_t event_start, 
                          cudaEvent_t event_stop, unsigned nsamp)
{
    SURVEY *survey = params -> survey;
    unsigned tid = params -> thread_num;
    float timestamp;

    cudaEventRecord(event_start, 0);	

    // Choose downsampling kernel depending on decimation factor
    if (survey -> downsample <= 32)
    {
	    dim3 gridDim(survey -> nchans, BEAMS);  
        downsample<<< gridDim, 128 >>> (input, output, nsamp, survey -> nchans, BEAMS, survey -> downsample);
    }
    else
    {
        dim3 gridDim((nsamp / survey -> downsample), survey -> nchans, survey -> nbeams);  
        downsample_atomics <<< gridDim, survey -> downsample, survey -> downsample * sizeof(float) / 32 >>> 
                              (input, output, nsamp, survey -> nchans, survey -> nbeams, survey -> downsample);
    }
    
    // All processing ready, wait for kernel execution
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    CudaCheckError();
    printf("%d: Performed Downsampling [Thread %d]: %lf\n", (int) (time(NULL) - params -> start), 
                                                              tid, timestamp);
}

// Wrapper for channelisation kernel
void channelise(cufftComplex *input, float* fir, cufftComplex *lagged_buffer, 
                THREAD_PARAMS* params, cudaEvent_t event_start,
                cudaEvent_t event_stop, cufftHandle *plan, unsigned nsamp)
{
    SURVEY *survey = params -> survey;
    unsigned tid = params -> thread_num;
    float timestamp;

    unsigned nbeams = survey -> nbeams;
    unsigned nsubs  = survey -> nchans;
    unsigned nchans = survey -> subchannels;

    // If PFB is enabled, apply filter
    if (survey -> apply_pfb)
    {
        cudaEventRecord(event_start, 0);
        unsigned num_threads = PFB_THREADS;
        dim3 grid(nsubs, nbeams);

        ppf_fir<<<grid, num_threads, NTAPS * num_threads * sizeof(float)>>>
                                (input, lagged_buffer, fir, nsamp / nchans, nsubs, nbeams, nchans);                                                              

        cudaThreadSynchronize();
        cudaEventRecord(event_stop, 0);
        cudaEventSynchronize(event_stop);
        cudaEventElapsedTime(&timestamp, event_start, event_stop);
        CudaCheckError();
        printf("%d: Performed PFB Filtering [Thread %d]: %lf\n", (int) (time(NULL) - params -> start), 
                                                                  tid, timestamp);
    }
    
    // Apply forward C2C FFTs
    cudaEventRecord(event_start, 0);
    for (unsigned i = 0; i < survey -> nbeams; i++)
        CufftSafeCall(cufftExecC2C(*plan, input + i * survey -> nchans * nsamp, 
                                     input + i * survey -> nchans * nsamp, 
                                     CUFFT_FORWARD));
    cudaThreadSynchronize();

    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    CudaCheckError();
    printf("%d: Performed FFT Channelisation [Thread %d]: %lf\n", (int) (time(NULL) - params -> start), 
                                                              tid, timestamp);	
}

// Fix data format in GPU memory after channelisation
// TODO: This should remove half of the FFT output (reflection)
void fix_channelisation_order(float2 *input, float* output, THREAD_PARAMS* params, cudaEvent_t event_start,
                        cudaEvent_t event_stop, unsigned nsamp)
{
    SURVEY *survey = params -> survey;
    unsigned tid = params -> thread_num;
    float timestamp;

    cudaEventRecord(event_start, 0);	

	dim3 gridDim(nsamp / survey -> subchannels, 
                 survey -> stop_channel - survey -> start_channel, 
                 survey -> nbeams);  
    fix_channelisation<<< gridDim, survey -> subchannels >>> 
                      (input, output, nsamp, survey -> nchans, BEAMS, survey -> subchannels, survey -> start_channel);

    // All processing ready, wait for kernel execution
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    CudaCheckError();
    printf("%d: Performed Channel Rearrangement [Thread %d]: %lf\n", (int) (time(NULL) - params -> start), 
                                                              tid, timestamp);
}

// =================================== CUDA CPU THREAD MAIN FUNCTION ====================================
void* run_beamformer(void* thread_params)
{
    THREAD_PARAMS* params = (THREAD_PARAMS *) thread_params;
    SURVEY *survey = params -> survey;
    int i, nsamp = params -> survey -> nsamp, nchans = params -> survey -> nchans, nants = survey -> nantennas;
    int loop_counter = 0, iters = 0, tid = params -> thread_num;
    time_t start = params -> start;

    printf("%d: Started thread %d\n", (int) (time(NULL) - start), tid);

    // Define range of beams which will be processed by thread
    unsigned nbeams = survey -> nbeams / survey -> num_threads;

    // Check if number of beams conforms to object macro declarations
    if (nbeams != BEAMS || BEAMS < BEAMS_PER_TB)
    {
        fprintf(stderr, "Incorrect beamformer configuration (BPT: %d, BEAMS: %d, beams: %d)\n", BEAMS_PER_TB, BEAMS, nbeams);
        exit(1);
    }

    // Check if number of taps conforms to object macro declaration
    if (survey -> ntaps != NTAPS)
    {
        fprintf(stderr, "Incorrect PFB configuration (TAPS: %d, taps: %d)\n", NTAPS, survey -> ntaps);
        exit(1);
    }

    // Initialise device
    CudaSafeCall(cudaSetDevice(params -> device_id));
    CudaSafeCall(cudaDeviceReset());
    cudaSetDeviceFlags(cudaDeviceBlockingSync);

    // Allocate output buffer
    float *d_output;
    CudaSafeCall(cudaMalloc((void **) &d_output, params -> outputsize));

    // Allocate input buffer
    unsigned char *d_input;
    CudaSafeCall(cudaMalloc((void **) &d_input, params -> outputsize));

    // Allocate shift buffers (only GPU's primary thread can use this memory buffer)
    float2 *beamshifts, *d_beamshifts;
    CudaSafeCall(cudaMallocHost((void **) &beamshifts, nants * nchans * nbeams * sizeof(float2), cudaHostAllocPortable));
    CudaSafeCall(cudaMalloc((void **) &d_beamshifts, nants * nchans * nbeams * sizeof(float2)));
    CudaSafeCall(cudaMemset(d_beamshifts, 0, nants * nchans * nbeams * sizeof(float2))); 

    // Initialise events / performance timers
    cudaEvent_t event_start, event_stop;
    float timestamp;
    cudaEventCreate(&event_start);

    // Initialise channelisation if required
    cuComplex *d_lagged_buffer;  // Buffer inter-iteration values
    float *d_fir;
    cufftHandle fft_plan;
    if (survey -> perform_channelisation)
    {

        printf("%d. Initialising PFB. Extra memory required: %.2f MB\n", (int) (time(NULL) - start), 
                nbeams * nchans * survey -> subchannels * NTAPS * sizeof(cuComplex) / (1024.0 * 1024.0));

        // Initialise cuFFT plan
        CufftSafeCall(cufftPlan1d(&fft_plan, survey -> subchannels, CUFFT_C2C, 
                                  survey -> nchans * nsamp / survey -> subchannels));

        // Allocate memory for channel buffers and FIR weights
        CudaSafeCall(cudaMalloc((void **) &d_fir, nchans * NTAPS * sizeof(float)));
        CudaSafeCall(cudaMalloc((void **) &d_lagged_buffer,  nbeams * nchans * survey -> subchannels * NTAPS * sizeof(cuComplex)));

        // Initialiase buffer to 1
        CudaSafeCall(cudaMemset(d_lagged_buffer, 1,  nbeams * nchans * survey -> subchannels * NTAPS * sizeof(cuComplex)));
        
        // Read fir coefficient file
        float *weights = (float *) malloc(NTAPS * survey -> subchannels * sizeof(float));
        char filename[500], temp[10];
        strcpy(filename, survey -> fir_path);
        strcat(filename, "/coeff_");
        sprintf(temp, "%d", survey -> ntaps);
        strcat(filename, temp);
        strcat(filename, "_");
        sprintf(temp, "%d", survey -> subchannels);
        strcat(filename, temp);
        strcat(filename, ".dat");

        printf("%d. Loading PFB coefficients file [%s]\n", (int) (time(NULL) - start), filename);
        FILE *fp = fopen(filename, "rb");
        if (fread(weights, sizeof(float), NTAPS * nchans, fp) <= 0)
        {
            fprintf(stderr, "Error reading PFB coefficients file. Exiting.\n");
            exit(1);
        }
        
        // Copy weights to GPU memory
        CudaSafeCall(cudaMemcpy(d_fir, weights, survey -> subchannels * NTAPS * sizeof(float), cudaMemcpyHostToDevice));

        // Free up stuff 
        free(weights);
        fclose(fp);
    }

 	// Initialise beamformer lookup table
    signed char table[16] = { 0 };
    table[0]  = 0;  table[1]   = 1;  table[2]   = 2;  table[3]  = 3;
    table[4]  = 4;  table[5]   = 5;  table[6]   = 6;  table[7]  = 7;
    table[8]  = -8;  table[9]  = -7; table[10]  = -6; table[11] = -5;
    table[12] = -4; table[13]  = -3; table[14]  = -2; table[15] = -1;
	CudaSafeCall(cudaMemcpyToSymbol(lookup_table, table, 16 * sizeof(signed char)));
    
    // Blocking sync when waiting for kernel launches
    cudaEventCreateWithFlags(&event_stop, cudaEventBlockingSync); 

    // Thread processing loop
    while (1)
    {
        // =========================== COPY INPUT DATA ==========================

        // Start recording CPU-GPU IO time
        cudaEventRecord(event_start, 0);

        if (loop_counter > params -> iterations)
        {
            // Copy input data to GPU memory
            CudaSafeCall(cudaMemcpyAsync(d_input, params -> input, nsamp * nchans 
                                        * nants * sizeof(unsigned char), cudaMemcpyHostToDevice));

            // Update beamformer shifts
            CudaSafeCall(cudaMemcpyAsync(d_beamshifts, 
                                         survey -> beam_shifts, 
                                         survey -> nantennas * survey -> nchans * nbeams * sizeof(float2), 
                                         cudaMemcpyHostToDevice));

            CudaSafeCall(cudaThreadSynchronize()); // Wait for all copies to finish            

            cudaEventRecord(event_stop, 0);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&timestamp, event_start, event_stop);
            printf("%d: Copied data to GPU [Beam %d]: %f\n", (int) (time(NULL) - start), tid, timestamp);
        }

        // Wait input barrier
        int ret = pthread_barrier_wait(params -> input_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
            { fprintf(stderr, "Error during barrier synchronisation 1 [thread]\n"); exit(0); }

        // ========================== GPU PROCESSING ===========================

        if (loop_counter > params -> iterations)
        {
            // Rearrange data to match beamformer processing requirements
            perform_rearrangement(d_input, (unsigned char *) d_output, params, event_start, event_stop, nsamp);

            // Copy rearranged data back to input buffer for conformity with future processing
            CudaSafeCall(cudaMemcpy(d_input, d_output, nchans * nsamp * ANTS * sizeof(unsigned char), cudaMemcpyDeviceToDevice));

            // Processing flow if channelisation is required
            if (survey -> perform_channelisation)
            {               
                // Perform beamforming (complex output)
                perform_beamforming(d_input, d_output, d_beamshifts, params, event_start, event_stop, nsamp);

                // Perform channelisation
                channelise((cufftComplex *) d_output, d_fir, d_lagged_buffer, 
                           params, event_start, event_stop, &fft_plan, nsamp);

                // Select required channel, fix channel ordering and transpose data
                fix_channelisation_order((float2 *) d_output, (float *) d_input, params, event_start, event_stop, nsamp);
            }
            else
            {
                // Perform beamforming (real output)
                perform_beamforming(d_input, d_output, d_beamshifts, params, event_start, event_stop, nsamp);

                // Downfactor (if required) and transpose data in GPU memory
                perform_downsampling(d_output, (float *) d_input, params, event_start, event_stop, nsamp);
            }
        }
              
        // Wait output barrier
        ret = pthread_barrier_wait(params -> output_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
            { fprintf(stderr, "Error during barrier synchronisation 2 [thread]\n"); exit(0); }

        // =========================  COPY OUTPUT DATA =========================
        if (loop_counter > params -> iterations) 
        { 
            // Collect and write output to host memory
            cudaEventRecord(event_start, 0);

            if (survey -> perform_channelisation)
            {
                unsigned nchans = survey -> subchannels * (survey -> stop_channel - survey -> start_channel) / 2;

                CudaSafeCall(cudaMemcpy( params -> output[params -> thread_num], d_input, 
                						 BEAMS * nchans * sizeof(float) * survey -> nsamp / survey -> subchannels,
                                         cudaMemcpyDefault));
            }
            else
                CudaSafeCall(cudaMemcpy( params -> output[params -> thread_num], d_input, 
                						 BEAMS * survey -> nchans * sizeof(float) * survey -> nsamp / survey -> downsample,
                                         cudaMemcpyDefault));


            cudaEventRecord(event_stop, 0);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&timestamp, event_start, event_stop);
            printf("%d: Copied data from GPU [Beam %d]: %f\n", 
                   (int) (time(NULL) - start), tid, timestamp);
        }

        // ======================== ITERATION HANDLING CODE ====================

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
