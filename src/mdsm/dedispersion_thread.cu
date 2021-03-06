#include "dedispersion_kernel.cu"
#include "dedispersion_thread.h"

DEVICES* initialise_devices(SURVEY* survey)
{
	int num_devices;

    // Enumerate devices and create DEVICE_INFO list, storing device capabilities
    cutilSafeCall(cudaGetDeviceCount(&num_devices));

    if (num_devices <= 0)
        { fprintf(stderr, "No CUDA-capable device found"); exit(0); }

    // Create and populate devices object
    DEVICES* devices = (DEVICES *) malloc(sizeof(DEVICES));
    devices -> devices = (DEVICE_INFO *) malloc(num_devices * sizeof(DEVICE_INFO));
    devices -> num_devices = 0;
    devices -> minTotalGlobalMem = (1024 * 1024 * 16);

    int orig_num = num_devices, counter = 0;
    char useDevice = 0;
    for(int i = 0; i < orig_num; i++) {
        cudaDeviceProp deviceProp;
        cutilSafeCall(cudaGetDeviceProperties(&deviceProp, i));
        useDevice = 0;
        
        if (deviceProp.major == 9999 && deviceProp.minor == 9999)
            { fprintf(stderr, "No CUDA-capable device found"); exit(0); }
        else if (deviceProp.totalGlobalMem / 1024 > 1024 * 2.5 * 1024) {

            // Check if device is in user specfied list, if any
            if (survey -> gpu_ids != NULL) {
                for(unsigned j = 0; j < survey -> num_gpus; j++)
                    if ((survey -> gpu_ids)[j] == i)
                        useDevice = 1;
            }
            else
                useDevice = 1;

            if (useDevice) {
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
        { fprintf(stderr, "No CUDA-capable device found"); exit(0); }

    return devices;
}

void level_one_cache_with_accumulators_brute_force(float *d_input, float *d_output, THREAD_PARAMS* params, cudaEvent_t event_start, cudaEvent_t event_stop, int maxshift)
{
    
    SURVEY *survey = params -> survey;

    int num_reg         = NUMREG;
    int divisions_in_t  = DIVINT;
    int divisions_in_dm = DIVINDM;
    int num_blocks_t    = (survey -> nsamp/(divisions_in_t * num_reg));
    int num_blocks_dm   = survey -> tdms / divisions_in_dm;

    float timestamp;
    float startdm = survey -> lowdm + survey -> dmstep * survey -> tdms / survey -> num_threads * params -> thread_num;
       
    dim3 threads_per_block(divisions_in_t, divisions_in_dm);
    dim3 num_blocks(num_blocks_t,num_blocks_dm); 

    cudaEventRecord(event_start, 0);	

    global_for_time_dedisperse_loop<<< num_blocks, threads_per_block >>>
    			(d_output, d_input, survey -> nsamp, survey -> nchans,
    			 startdm/survey -> tsamp, survey
			 -> dmstep/survey -> tsamp,
			 maxshift);

    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("%d: Performed Brute-Force Dedispersion %d: %lf\n", (int) (time(NULL) - params -> start), params -> thread_num, timestamp);

}

// Perform subband dedispersion
void subband_dedispersion(float *d_input, float *d_output, THREAD_PARAMS* params, cudaEvent_t event_start, cudaEvent_t event_stop)
{
	// Declare function variables
    int maxshift = params -> maxshift, tid = params -> thread_num, num_threads = params -> num_threads;
    int i, j, nsamp = params -> survey -> nsamp, nchans = params -> survey -> nchans;
    float tsamp = params -> survey -> tsamp;
    SURVEY *survey = params -> survey;
    time_t start = params -> start;
    float timestamp;

    // Define kernel thread configuration
    int blocksize_dedisp = 128; // gridsize_dedisp = 128, 
    dim3 gridDim_bin(128, (nchans / 128.0) < 1 ? 1 : nchans / 128.0);
    dim3 blockDim_bin(min(nchans, 128), 1);

    // Survey parameters
    int lobin = survey -> pass_parameters[0].binsize;
    int binsize, inshift, outshift, kernelBin;

	// ------------------------------------- Perform downsampling on GPU --------------------------------------
	// All input data is copied to all GPUs, so we need to perform binning on all of them

	cudaEventRecord(event_start, 0);
	binsize = lobin; inshift = 0, outshift = 0, kernelBin = binsize;
	for( i = 0; i < survey -> num_passes; i++) {

		if (binsize != 1) {        // if binsize is 1, no need to perform binning
			if (i == 0) {          // Original raw data not required, special case
				inplace_binning_kernel<<< gridDim_bin, blockDim_bin >>>(d_input, nsamp + maxshift, nchans, kernelBin);
				inplace_memory_reorganisation<<< gridDim_bin, blockDim_bin >>>(d_input, nsamp + maxshift, nchans, kernelBin);
				cutilSafeCall( cudaMemset(d_input + (nsamp + maxshift) * nchans / binsize, 0,
										 ((nsamp + maxshift) * nchans - (nsamp + maxshift) * nchans / binsize) * sizeof(float)));
			} else {
				inshift = outshift;
				outshift += ( (nsamp + maxshift) * nchans) * 2 / binsize;
				binning_kernel<<< gridDim_bin, blockDim_bin >>>(d_input, (nsamp + maxshift) * 2 / binsize,
																nchans, kernelBin, inshift, outshift);
			}
		}

		binsize *= 2;
		kernelBin = 2;
	}

	cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("%d: Processed Binning %d: %lf\n", (int) (time(NULL) - start), tid, timestamp);

	// --------------------------------- Perform subband dedispersion on GPU ---------------------------------
	cudaEventRecord(event_start, 0);

	// Handle dedispersion maxshift
	inshift = 0, outshift = 0;
	int ncalls, tempval = (int) (params -> dmshifts[(survey -> nsubs - 1) * nchans / survey -> nsubs]
						  * survey -> pass_parameters[survey -> num_passes - 1].highdm /
						  survey -> tsamp );
	float startdm;

	for( i = 0; i < survey -> num_passes; i++) {

		// Setup call parameters (ncalls is split among all GPUs)
		binsize = survey -> pass_parameters[i].binsize;
		ncalls = survey -> pass_parameters[i].ncalls / num_threads;
		startdm = survey -> pass_parameters[i].lowdm + survey -> pass_parameters[i].sub_dmstep * ncalls * tid;

		// Perform subband dedispersion
		opt_dedisperse_subband <<< dim3((nsamp + tempval) / binsize / blocksize_dedisp, ncalls), 
                                   blocksize_dedisp >>>
			    (d_output, d_input, (nsamp + tempval) / binsize, nchans, survey -> nsubs,
			     startdm, survey -> pass_parameters[i].sub_dmstep,
			     tsamp * binsize, maxshift - tempval, inshift, outshift);

		outshift += (nsamp + tempval) * survey -> nsubs * ncalls / binsize ;
		inshift += (nsamp + maxshift) * nchans / binsize;
	}

	cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("%d: Processed Subband Dedispersion %d: %lf\n", (int) (time(NULL) - start), tid, timestamp);

	// Copy subband output as dedispersion input
	cutilSafeCall( cudaMemcpy(d_input, d_output, params -> outputsize, cudaMemcpyDeviceToDevice) );

	// ------------------------------------- Perform dedispersion on GPU --------------------------------------
	cudaEventRecord(event_start, 0);

	float dm = 0.0;
	inshift = outshift = 0;
	for (i = 0; i < survey -> num_passes; i++) {

		// Setup call parameters (ncalls is split among all GPUs)
		ncalls = survey -> pass_parameters[i].ncalls / num_threads;
		startdm = survey -> pass_parameters[i].lowdm + survey -> pass_parameters[i].sub_dmstep * ncalls * tid;
		binsize = survey -> pass_parameters[i].binsize;

		// Perform subband dedispersion for all subband calls
		for(j = 0; j < ncalls; j++) {

			dm = max(startdm + survey -> pass_parameters[i].sub_dmstep * j
				 - survey -> pass_parameters[i].calldms * survey -> pass_parameters[i].dmstep / 2, 0.0);

			opt_dedisperse_loop<<< dim3(nsamp / blocksize_dedisp, survey -> pass_parameters[i].calldms), 
                                   blocksize_dedisp, blocksize_dedisp >>>
				(d_output, d_input, nsamp / binsize, survey -> nsubs,
				 tsamp * binsize, nchans /  survey -> nsubs,
				 dm, survey -> pass_parameters[i].dmstep, tempval / binsize,
	  			 inshift, outshift);

			inshift += (nsamp + tempval) * survey -> nsubs / binsize;
			outshift += nsamp * survey -> pass_parameters[i].calldms / binsize;
		}
	}

	cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);

	printf("%d: Processed Deispersion %d: %lf\n", (int) (time(NULL) - start), tid, timestamp);
}

// Perform brute-froce dedisperion
void brute_force_dedispersion(float *d_input, float *d_output, THREAD_PARAMS* params, cudaEvent_t event_start, cudaEvent_t event_stop, int maxshift)
{
	// Define function variables;
    SURVEY *survey = params -> survey;
    float timestamp;

    // ------------------------------------- Perform dedispersion on GPU --------------------------------------
    cudaEventRecord(event_start, 0);

    float startdm = survey -> lowdm + survey -> dmstep * survey -> tdms / survey -> num_threads * params -> thread_num;
  
    // Optimised kernel
    opt_dedisperse_loop<<< dim3(survey -> nsamp / 128, survey -> tdms / survey -> num_threads), 128 >>>
			(d_output, d_input, survey -> nsamp, survey -> nchans,
			 survey -> tsamp, 1, startdm, survey -> dmstep, maxshift, 0, 0);


    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("%d: Performed Brute-Force Dedispersion %d: %lf\n", (int) (time(NULL) - params -> start),
															   params -> thread_num, timestamp);

}

// Dedispersion algorithm
void* dedisperse(void* thread_params)
{
    THREAD_PARAMS* params = (THREAD_PARAMS *) thread_params;
    int i, nsamp = params -> survey -> nsamp, nchans = params -> survey -> nchans;
    int ret, loop_counter = 0, maxshift = params -> maxshift, iters = 0, tid = params -> thread_num;
    time_t start = params -> start;
    SURVEY *survey = params -> survey;
    float *d_input, *d_output;

    printf("%d: Started thread %d\n", (int) (time(NULL) - start), tid);

    // Initialise device, allocate device memory and copy dmshifts and dmvalues to constant memory
    cutilSafeCall( cudaSetDevice(params -> device_id));
    cudaSetDeviceFlags( cudaDeviceBlockingSync );

    cutilSafeCall( cudaMalloc((void **) &d_input, params -> inputsize));
    cutilSafeCall( cudaMalloc((void **) &d_output, params -> outputsize));
    cutilSafeCall( cudaMemcpyToSymbol(dm_shifts, params -> dmshifts, nchans * sizeof(nchans)) );

    // Temporary store for maxshift
    float *tempshift = (float *) malloc(maxshift * nchans * sizeof(float));
    float *tempshift2 = (float *) malloc(maxshift * nchans * sizeof(float));

    // Initialise events / performance timers
    cudaEvent_t event_start, event_stop;
    float timestamp;

    cudaEventCreate(&event_start);
    cudaEventCreateWithFlags(&event_stop, cudaEventBlockingSync); // Blocking sync when waiting for kernel launches

    // Thread processing loop
    while (1) {

        if (loop_counter >= params -> iterations) {

            // Read input data into GPU memory
            cudaEventRecord(event_start, 0);
            if (loop_counter == 1) {
                // First iteration, no available extra samples, so load everything to GPU memory
                cutilSafeCall( cudaMemcpy(d_input, params -> input, (nsamp + maxshift) * nchans * sizeof(float), cudaMemcpyHostToDevice) );

                // Keep a copy of maxshift in memory
                for(i = 0; i < nchans; i++)
                    memcpy(tempshift + (maxshift * i), params -> input + i * (nsamp + maxshift) + nsamp, maxshift * sizeof(float)); // NOTE: Optimise
            }
            else {
                // Copy previous maxshift to input buffer
                for(i = 0; i < nchans; i++)
                    memcpy(params -> input + i * (nsamp + maxshift), tempshift + maxshift * i, maxshift * sizeof(float)); // NOTE: Optimise

               // cutilSafeCall( cudaMemcpy(d_input, tempshift, maxshift * nchans * sizeof(float), cudaMemcpyHostToDevice) );
                cutilSafeCall( cudaMemcpy(d_input, params -> input,
                                          (nsamp + maxshift) * nchans * sizeof(float), cudaMemcpyHostToDevice) );

                // Keep a copy of maxshift in memory
                for(i = 0; i < nchans; i++)
                    memcpy(tempshift + (maxshift * i), params -> input + i * (nsamp + maxshift) + nsamp, maxshift * sizeof(float)); // NOTE: Optimise
            }

//            if (loop_counter == 1) {
//                // First iteration, no available extra samples, so load everything to GPU memory
//                cutilSafeCall( cudaMemcpy(d_input, params -> input, (nsamp + maxshift) * nchans * sizeof(float), cudaMemcpyHostToDevice) );
//                memcpy(tempshift, params -> input + nsamp * nchans, maxshift * nchans * sizeof(float)); // NOTE: Optimise
//            }
//            else {
//                // Shift buffers and load input buffer
//                cutilSafeCall( cudaMemcpy(d_input, tempshift, maxshift * nchans * sizeof(float), cudaMemcpyHostToDevice) );
//                cutilSafeCall( cudaMemcpy(d_input + (maxshift * nchans), params -> input,
//                                          nsamp * nchans * sizeof(float), cudaMemcpyHostToDevice) );
//                memcpy(tempshift, params -> input + (nsamp - maxshift) * nchans, maxshift * nchans * sizeof(float)); // NOTE: Optimise
//            }

            cudaEventRecord(event_stop, 0);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&timestamp, event_start, event_stop);
            printf("%d: Copied data to GPU %d: %f\n", (int) (time(NULL) - start), tid, timestamp);

            // Clear GPU output buffer
            cutilSafeCall( cudaMemset(d_output, 0, params -> outputsize));
        }

        // Wait input barrier
        ret = pthread_barrier_wait(params -> input_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
            { fprintf(stderr, "Error during barrier synchronisation 1 [thread]\n"); exit(0); }

        if (loop_counter >= params -> iterations){
        	if (survey -> useBruteForce){
			if(survey -> useL1Cache){
				level_one_cache_with_accumulators_brute_force(d_input,d_output, params, event_start, event_stop, maxshift);
			} else {
		        	brute_force_dedispersion(d_input,d_output, params, event_start, event_stop, maxshift);
			}
		}
		else
        		subband_dedispersion(d_input, d_output, params, event_start, event_stop);
        }

        // Wait output barrier
        ret = pthread_barrier_wait(params -> output_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
            { fprintf(stderr, "Error during barrier synchronisation 2 [thread]\n"); exit(0); }

        if(loop_counter >= params -> iterations) { 

            // Collect and write output to host memory
            cudaEventRecord(event_start, 0);
            cutilSafeCall(cudaMemcpy( params -> output, d_output, 
            						  params -> dedispersed_size * sizeof(float),
                                      cudaMemcpyDeviceToHost) );
            cudaEventRecord(event_stop, 0);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&timestamp, event_start, event_stop);
        }

        // Acquire rw lock
        if (pthread_rwlock_rdlock(params -> rw_lock))
            { fprintf(stderr, "Unable to acquire rw_lock [thread]\n"); exit(0); }

        // Update params  
        nsamp = params -> survey -> nsamp;

        // Stopping clause
        if (((THREAD_PARAMS *) thread_params) -> stop) {

            if (iters >= params -> iterations - 1) {  

                // Release rw_lock
                if (pthread_rwlock_unlock(params -> rw_lock))
                    { fprintf(stderr, "Error releasing rw_lock [thread]\n"); exit(0); }

                for(i = 0; i < params -> maxiters - params -> iterations; i++) {
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

    cutilSafeCall( cudaFree(d_output));
    cutilSafeCall( cudaFree(d_input));
    cudaEventDestroy(event_stop);
    cudaEventDestroy(event_start); 

    printf("%d: Exited gracefully %d\n", (int) (time(NULL) - start), tid);
    pthread_exit((void*) thread_params);
}
