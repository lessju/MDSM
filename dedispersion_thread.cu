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

    if (*num_devices == 0)
        { fprintf(stderr,"No CUDA-capable device found"); exit(0); }

    *num_devices = 1;
 
    // OPTIONAL: Perform load-balancing calculations
    return info;
}

// Dedispersion algorithm
void* dedisperse(void* thread_params)
{
    THREAD_PARAMS* params = (THREAD_PARAMS *) thread_params;
    int i, j, nsamp = params -> survey -> nsamp, nchans = params -> survey -> nchans;
    int ret, loop_counter = 0, maxshift = params -> maxshift, iters = 0;
    time_t start = params -> start;
    SURVEY *survey = params -> survey;
    float *d_input, *d_output;      

    printf("%d: Started thread %d\n", (int) (time(NULL) - start), params -> thread_num);

    // Initialise device, allocate device memory and copy dmshifts and dmvalues to constant memory
    cutilSafeCall( cudaSetDevice(params -> device_id));

    cutilSafeCall( cudaMalloc((void **) &d_input, params -> inputsize));
    cutilSafeCall( cudaMalloc((void **) &d_output, params -> outputsize));
    cutilSafeCall( cudaMemcpyToSymbol(dm_shifts, params -> dmshifts, params -> nchans * sizeof(float)) );

    // Temporary store for maxshift
    float *tempshift = (float *) malloc(maxshift * nchans * sizeof(float));

    // Initialise events / performance timers
    cudaEvent_t event_start, event_stop;
    float timestamp;

    cudaEventCreate(&event_start); 
    cudaEventCreate(&event_stop); 

    // Define kernel thread configuration
    int gridsize_dedisp = 128, blocksize_dedisp = 128;
    dim3 gridDim_bin(128, nchans / 128.0 < 1 ? 1 : nchans / 128.0);
    dim3 blockDim_bin(min(nchans, 128), 1);

    // Survey parameters
    int lobin = survey -> pass_parameters[0].binsize;
    int binsize, inshift, outshift, kernelBin;

    // Calculate output size
    int outsize = 0;

    for(i = 0; i < survey -> num_passes; i++)
        outsize += survey -> pass_parameters[i].ndms / survey -> pass_parameters[i].binsize;
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
            printf("%d: Copied data to GPU %d: %f\n", (int) (time(NULL) - start), params -> thread_num, timestamp);

            // Clear GPU output buffer
            cutilSafeCall( cudaMemset(d_output, 0, params -> outputsize));
        }

        // Wait input barrier
        ret = pthread_barrier_wait(params -> input_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
            { fprintf(stderr, "Error during barrier synchronisation 1 [thread]\n"); exit(0); }

        if (loop_counter >= params -> iterations) {

            // ------------------------------------- Perform downsampling on GPU --------------------------------------
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
            printf("%d: Processed Binning %d: %lf\n", (int) (time(NULL) - start), params -> thread_num, timestamp);

           // CHECK OUTPUT
/*            cutilSafeCall(cudaMemcpy( params -> output, d_input, params -> outputsize, cudaMemcpyDeviceToHost) );
            for(i = 0; i < (params -> nsamp + params -> maxshift) * params -> nchans / params -> survey -> pass_parameters[0].binsize; i++)
                if (params -> output[i] > 256)
                    printf("bin.. %d, %f\n", i, params -> output[i]);
*/
            // --------------------------------- Perform subband dedispersion on GPU ---------------------------------
            cudaEventRecord(event_start, 0);

            // Handle dedispersion maxshift
            binsize = lobin; inshift = 0, outshift = 0;
            int tempval = (int) (params -> dmshifts[(survey -> nsubs - 1) * nchans / survey -> nsubs]
                             * survey -> pass_parameters[survey -> num_passes - 1].highdm /  
                             survey -> tsamp );

            for( i = 0; i < survey -> num_passes; i++) {

                // Perform subband dedispersion
                dedisperse_subband <<< dim3(gridsize_dedisp, survey -> pass_parameters[i].ncalls), blocksize_dedisp >>>
                    (d_output, d_input, (nsamp + tempval) / binsize, nchans, survey -> nsubs, 
                     survey -> pass_parameters[i].lowdm, survey -> pass_parameters[i].sub_dmstep,
                     params -> tsamp * binsize, inshift, outshift); 

                inshift += (nsamp + maxshift) * nchans / binsize;
                outshift += (nsamp + tempval) * survey -> nsubs * survey -> pass_parameters[i].ncalls / binsize ;
                binsize *= 2;
            }

            cudaEventRecord(event_stop, 0);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&timestamp, event_start, event_stop);
            printf("%d: Processed Subband Dedispersion %d: %lf\n", (int) (time(NULL) - start), params -> thread_num, timestamp);     

            // CHECK OUTPUT
/*            cutilSafeCall(cudaMemcpy( params -> output, d_output, params -> outputsize, cudaMemcpyDeviceToHost) );

            int a, b, c, l, k;
            inshift = 0;
            for(i = 0; i < survey -> num_passes; i++) {

                a = (nsamp + tempval) / survey -> pass_parameters[i].binsize;
                b = survey -> nsubs;
                c = survey -> pass_parameters[i].ncalls;

                for (l = 0; l < c; l++)          //ncalls
                    for (j = 0; j < b; j++)      //nsubs
                        for (k = 0; k < a; k++)  //nsamp
                            if (params -> output[inshift + l*a*b + j*a + k] > 6 * 1024)
                                {    printf("%d. ncall: %d, nsub: %d, samp: %d, %f\n", i, l, j, k, params -> output[inshift + l*a*b + j*a + k]);   } 

                inshift = a * b * c;
            }
*/
            // Copy subband output as dedispersion input
            cutilSafeCall( cudaMemcpy(d_input, d_output, params -> outputsize, cudaMemcpyDeviceToDevice) );

            // ------------------------------------- Perform dedispersion on GPU --------------------------------------
            cudaEventRecord(event_start, 0);

            float dm = 0.0;
            binsize = lobin, inshift = outshift = 0;
            for (i = 0; i < survey -> num_passes; i++) {

                // Perform subband dedispersion for all subband calls
                for(j = 0; j < survey -> pass_parameters[i].ncalls; j++) {

                    dm = max(survey -> pass_parameters[i].lowdm + survey -> pass_parameters[i].sub_dmstep * j
                         - survey -> pass_parameters[i].calldms * survey -> pass_parameters[i].dmstep / 2, 0.0);

                    dedisperse_loop<<< dim3(gridsize_dedisp, survey -> pass_parameters[i].calldms), blocksize_dedisp >>>
                        (d_output, d_input, nsamp / binsize, survey -> nsubs, 
                         params -> tsamp * binsize, params -> nchans /  survey -> nsubs, 
                         dm, survey -> pass_parameters[i].dmstep,
                         inshift, outshift);
                    
                    inshift += (nsamp + tempval) * survey -> nsubs / binsize;
                    outshift += nsamp * survey -> pass_parameters[i].calldms / binsize;
                }

                binsize *= 2;
            }

            cudaEventRecord(event_stop, 0);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&timestamp, event_start, event_stop);

            // CHECK OUTPUT
/*            cutilSafeCall(cudaMemcpy( params -> output, d_output, params -> outputsize, cudaMemcpyDeviceToHost) );

            inshift = 0;
            for(i = 0; i < survey -> num_passes; i++) {

                a = survey -> nsamp / survey -> pass_parameters[i].binsize;
                b = survey -> pass_parameters[i].ndms;
                for (j = 0; j < b; j++)
                    for (k = 0; k < a; k++)
                        if (params -> output[inshift + j * a + k] < 6 * 1024)
                            {    printf("%d. ndm: %d, samp: %d, %d, %f\n", i, j, k, inshift + j * a + k, params -> output[inshift + j * a + k]);   } 

                inshift += a * b;
            }
*/
            printf("%d: Processed Dedispersion %d: %lf\n", (int) (time(NULL) - start), params -> thread_num, timestamp);
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
            printf("%d: Written output %d: %f\n", (int) (time(NULL) - start), params -> thread_num, timestamp);
        }

        // Acquire rw lock
        if (pthread_rwlock_rdlock(params -> rw_lock))
            { fprintf(stderr, "Unable to acquire rw_lock [thread]\n"); exit(0); }

        // Update params  
        nsamp = params -> nsamp;

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

    printf("%d: Exited gracefully %d\n", (int) (time(NULL) - start), params -> thread_num);
    pthread_exit((void*) thread_params);
}
