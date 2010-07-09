#include "dedispersion_kernel.cu"
#include "dedispersion_thread.h"

DEVICE_INFO** initialise_devices(int *num_devices)
{
    // Enumerate devices and create DEVICE_INFO list, storing device capabilities
    cutilSafeCall(cudaGetDeviceCount(num_devices));

    if (*num_devices <= 0) 
        { fprintf(stderr, "No CUDA-capable device found"); exit(0); }

    DEVICE_INFO **info = (DEVICE_INFO **) malloc( *num_devices * sizeof(DEVICE_INFO *));

    int orig_num = *num_devices, counter = 0;
    for(int i = 0; i < orig_num; i++) {
        cudaDeviceProp deviceProp;
        cutilSafeCall(cudaGetDeviceProperties(&deviceProp, i));
        
        if (deviceProp.major == 9999 && deviceProp.minor == 9999)
            { fprintf(stderr, "No CUDA-capable device found"); exit(0); }
        else {
            if (deviceProp.totalGlobalMem < (long) 2 * 1024 * 1024 * 1024)
                *num_devices = *num_devices - 1;
            else {
                info[counter] = (DEVICE_INFO *) malloc(sizeof(DEVICE_INFO));
                info[counter] -> multiprocessor_count = deviceProp.multiProcessorCount;
                info[counter] -> constant_memory = deviceProp.totalConstMem;
                info[counter] -> shared_memory = deviceProp.sharedMemPerBlock;
                info[counter] -> register_count = deviceProp.regsPerBlock;
                info[counter] -> thread_count = deviceProp.maxThreadsPerBlock;
                info[counter] -> clock_rate = deviceProp.clockRate;
                info[counter] -> device_id = i;
                counter++;
            }
        }
    }

    *num_devices = 1;  // TEMPORARY TESTING HACK

    if (*num_devices == 0)
        { fprintf(stderr,"No CUDA-capable device found"); exit(0); }

    // OPTIONAL: Perform load-balancing calculations
    return info;
}

// Dedispersion algorithm
void* dedisperse(void* thread_params)
{
    THREAD_PARAMS* params = (THREAD_PARAMS *) thread_params;
    int i, j, nsamp = params -> survey -> nsamp, nchans = params -> survey -> nchans;
    float tsamp = params -> survey -> tsamp;
    int ret, loop_counter = 0, maxshift = params -> maxshift, iters = 0, tid = params -> thread_num;
    int num_threads = params -> num_threads;
    time_t start = params -> start;
    SURVEY *survey = params -> survey;
    float *d_input, *d_output;      

    printf("%d: Started thread %d\n", (int) (time(NULL) - start), tid);

    // Initialise device, allocate device memory and copy dmshifts and dmvalues to constant memory
    cutilSafeCall( cudaSetDevice(params -> device_id));
    cudaSetDeviceFlags( cudaDeviceBlockingSync );

    cutilSafeCall( cudaMalloc((void **) &d_input, params -> inputsize));
    cutilSafeCall( cudaMalloc((void **) &d_output, params -> outputsize));
    cutilSafeCall( cudaMemcpyToSymbol(dm_shifts, params -> dmshifts, nchans * sizeof(float)) );

    // Temporary store for maxshift
    float *tempshift = (float *) malloc(maxshift * nchans * sizeof(float));

    // Initialise events / performance timers
    cudaEvent_t event_start, event_stop;
    float timestamp;

    cudaEventCreate(&event_start); 
    cudaEventCreateWithFlags(&event_stop, cudaEventBlockingSync); // Blocking sync when waiting for kernel launches 

    // Define kernel thread configuration
    int gridsize_dedisp = 128, blocksize_dedisp = 128;
    dim3 gridDim_bin(128, (nchans / 256.0) < 1 ? 1 : nchans / 256.0);
    dim3 blockDim_bin(min(nchans, 256), 1);

    // Survey parameters
    int lobin = survey -> pass_parameters[0].binsize;
    int binsize, inshift, outshift, kernelBin;

    // Calculate output size and allocate thread output buffer
    int outsize = 0;
    for(i = 0; i < survey -> num_passes; i++)
        outsize += (survey -> pass_parameters[i].ncalls / num_threads) * survey -> pass_parameters[i].calldms 
                   / survey -> pass_parameters[i].binsize;
    outsize *= nsamp;

    // Thread processing loop
    while (1) {
       
        if (loop_counter >= params -> iterations) {

            // Read input data into GPU memory
            cudaEventRecord(event_start, 0);
            if (loop_counter == 1)
                // First iteration, no available extra samples, so load everything to GPU memory
                cutilSafeCall( cudaMemcpy(d_input, params -> input, (nsamp + maxshift) * nchans * sizeof(float), cudaMemcpyHostToDevice) );
            else {
                // Shift buffers and load input buffer
                cutilSafeCall( cudaMemcpy(d_input, tempshift, maxshift * nchans * sizeof(float), cudaMemcpyHostToDevice) );
                cutilSafeCall( cudaMemcpy(d_input + (maxshift * nchans), params -> input, 
                                          nsamp * nchans * sizeof(float), cudaMemcpyHostToDevice) );
            }

            // Copy maxshift value to temporary host store
            // TODO: Inefficient, find better way
            memcpy(tempshift, params -> input + nsamp * nchans, maxshift * nchans * sizeof(float));

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

        if (loop_counter >= params -> iterations) {

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
                dedisperse_subband <<< dim3(gridsize_dedisp, ncalls), blocksize_dedisp >>>
                    (d_output, d_input, (nsamp + tempval) / binsize, nchans, survey -> nsubs, 
                     startdm, survey -> pass_parameters[i].sub_dmstep,
                     tsamp * binsize, inshift, outshift); 

                inshift += (nsamp + maxshift) * nchans / binsize;
                outshift += (nsamp + tempval) * survey -> nsubs * ncalls / binsize ;
            }

            cudaEventRecord(event_stop, 0);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&timestamp, event_start, event_stop);
            printf("%d: Processed Subband Dedispersion %d: %lf\n", (int) (time(NULL) - start), tid, timestamp);     

            // CHECK OUTPUT
//            cutilSafeCall(cudaMemcpy( params -> output, d_output, params -> outputsize, cudaMemcpyDeviceToHost) );

//            int a, b, c, l, k;
//            inshift = 0;
//            for(i = 0; i < survey -> num_passes; i++) {

//                a = (nsamp + tempval) / survey -> pass_parameters[i].binsize;
//                b = survey -> nsubs;
//                c = survey -> pass_parameters[i].ncalls / num_threads;

//                for (l = 0; l < c; l++)          //ncalls
//                    for (j = 0; j < b; j++)      //nsubs
//                        for (k = 0; k < a; k++)  //nsamp
//                            if (params -> output[inshift + l*a*b + j*a + k] > 7 * 1024)
//                                {    printf("%d. ncall: %d, nsub: %d, samp: %d, %f\n", i, l, j, k, params -> output[inshift + l*a*b + j*a + k]);   } 

//                inshift = a * b * c;
//            }

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

                    dedisperse_loop<<< dim3(gridsize_dedisp, survey -> pass_parameters[i].calldms), blocksize_dedisp >>>
                        (d_output, d_input, nsamp / binsize, survey -> nsubs, 
                         tsamp * binsize, nchans /  survey -> nsubs, 
                         dm, survey -> pass_parameters[i].dmstep,
                         inshift, outshift);
                    
                    inshift += (nsamp + tempval) * survey -> nsubs / binsize;
                    outshift += nsamp * survey -> pass_parameters[i].calldms / binsize;
                }
            }

            cudaEventRecord(event_stop, 0);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&timestamp, event_start, event_stop);

            // CHECK OUTPUT
//            cutilSafeCall(cudaMemcpy( params -> output, d_output, params -> outputsize, cudaMemcpyDeviceToHost) );

//            inshift = 0;
//            for(i = 0; i < survey -> num_passes; i++) {

//                a = survey -> nsamp / survey -> pass_parameters[i].binsize;
//                b = (survey -> pass_parameters[i].ncalls / num_threads) * survey -> pass_parameters[i].calldms;
//                for (j = 0; j < b; j++)
//                    for (k = 0; k < a; k++)
//                        if (params -> output[inshift + j * a + k] < 6 * 1024)
//                            {    printf("%d.%d - ndm: %d, samp: %d, %d, %f\n", tid, i, j, k, inshift + j * a + k, params -> output[inshift + j * a + k]);   } 

//                inshift += a * b;
//            }

            printf("%d: Processed Deispersion %d: %lf\n", (int) (time(NULL) - start), tid, timestamp);
       }

        // Wait output barrier
        ret = pthread_barrier_wait(params -> output_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
            { fprintf(stderr, "Error during barrier synchronisation 2 [thread]\n"); exit(0); }

        if(loop_counter >= params -> iterations) { 

            // Collect and write output to host memory
            cudaEventRecord(event_start, 0);
            cutilSafeCall(cudaMemcpy( params -> output, d_output, 
                                      outsize * sizeof(float), 
                                      cudaMemcpyDeviceToHost) );
            cudaEventRecord(event_stop, 0);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&timestamp, event_start, event_stop);
            printf("%d: Written output %d: %f\n", (int) (time(NULL) - start), tid, timestamp);
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
