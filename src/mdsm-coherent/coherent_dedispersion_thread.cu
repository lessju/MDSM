#include "coherent_dedispersion_kernel.cu"
#include "coherent_dedispersion_thread.h"

DEVICES* initialise_devices(OBSERVATION* obs)
{
	int num_devices;

    // Enumerate devices and create DEVICE_INFO list, storing device capabilities
    cudaGetDeviceCount(&num_devices);

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
        cudaGetDeviceProperties(&deviceProp, i);
        useDevice = 0;
        
        if (deviceProp.major == 9999 && deviceProp.minor == 9999)
            { fprintf(stderr, "No CUDA-capable device found"); exit(0); }
        else if (deviceProp.totalGlobalMem / 1024 > 1024 * 2.5 * 1024) {

            // Check if device is in user specfied list, if any
            if (obs -> gpu_ids != NULL) {
                for(unsigned j = 0; j < obs -> num_gpus; j++)
                    if ((obs -> gpu_ids)[j] == i)
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

// -------------------------- Error checking function ------------------------
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "CUDA error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}

// -------------------------- Main processing loop --------------------------

// Main procesing loop
void* dedisperse(void* thread_params)
{
    THREAD_PARAMS* params = (THREAD_PARAMS *) thread_params;
    int i, j, nchans = params -> obs -> nchans;
    int ret, loop_counter = 0, iters = 0, tid = params -> thread_num;
    time_t start = params -> start;
    OBSERVATION *obs = params -> obs;

    unsigned gpuSamples  = obs -> gpuSamples;
    unsigned nsamp       = obs -> nsamp;
    unsigned fftsize     = obs -> fftsize;
    unsigned overlap     = obs -> overlap;
    unsigned numBlocks   = obs -> numBlocks;

    printf("%d: Started thread %d\n", (int) (time(NULL) - start), tid);

    // Initialise device, allocate device memory
    cudaSetDevice(params -> device_id);
    cudaSetDeviceFlags( cudaDeviceBlockingSync );

    float *device_profile;
    cufftComplex *device_idata, *device_tempFoldData; 
    cudaMalloc((void **) &device_idata, params -> device_isize);

    if (obs -> folding)
    {
        cudaMalloc((void **) &device_profile, params -> profile_size);
        cudaMalloc((void **) &device_tempFoldData, params -> profile_size * 2);
        cudaMemset(device_profile, 0, params -> profile_size);
    }

    // Temporary store for buffer shift (buffer overlays)
    cufftComplex *tempshift =  (cufftComplex *) malloc(obs -> wingLen * nchans * sizeof(cufftComplex));

    // Initialise events / performance timers
    cudaEvent_t event_start, event_stop;
    float timestamp;

    cudaEventCreate(&event_start);
    cudaEventCreateWithFlags(&event_stop, cudaEventBlockingSync); // Blocking sync

    // Create FFT plan
    cufftHandle plan, profile_fplan, profile_iplan;
    cufftPlan1d(&plan, fftsize, CUFFT_C2C, nchans * numBlocks);
    cufftPlan1d(&profile_fplan, obs -> profile_bins, CUFFT_R2C, nchans);
    cufftPlan1d(&profile_iplan, obs -> profile_bins, CUFFT_C2R, 1);
    unsigned blocksize = 512;

    double currTime = 0;

    // Thread processing loop
    while (1) 
    {
        if (loop_counter >= params -> iterations) 
        {
            // Read input data into GPU memory
            cudaEventRecord(event_start, 0);

            // If not the first iteration, then we need to copy the previous 
            // overlap to the input buffer    
            if (loop_counter > 1)
                for (i = 0; i < nchans; i++)
                    memcpy(params -> host_idata + i * gpuSamples,
                           tempshift + obs -> wingLen * i,
                           obs -> wingLen * sizeof(cufftComplex));

            // Copy input data to GPU memory
            // The +overlap for gpuSamples comes in in the last fftsize copy to GPU memory
            for(i = 0; i < nchans; i++)
                for(j = 0; j  < numBlocks; j++)
                    cudaMemcpy(device_idata + (i * numBlocks + j) * fftsize, 
                               params -> host_idata + (i * gpuSamples) + j * (fftsize - overlap), 
                               fftsize * sizeof(cufftComplex), 
                               cudaMemcpyHostToDevice);

            // Keep a copy of the last overlap samples in host memory
            for (i = 0; i < nchans; i++)
                memcpy(tempshift + obs -> wingLen * i, 
                       params -> host_idata + gpuSamples * i + gpuSamples - obs -> wingLen,
                       obs -> wingLen * sizeof(cufftComplex));

            cudaEventRecord(event_stop, 0);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&timestamp, event_start, event_stop);
            checkCUDAError("Copying data to GPU");
            printf("%d: Copied data to GPU %d: %f\n", (int) (time(NULL) - start), tid, timestamp);
        }

        // Wait input barrier
        ret = pthread_barrier_wait(params -> input_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
            { fprintf(stderr, "Error during barrier synchronisation 1 [thread]\n"); exit(0); }

        // GPU processing stage
        if (loop_counter >= params -> iterations)
        {
            // ---------------------- FFT all the channels in place ----------------------
//            cudaEventRecord(event_start, 0);
//            cufftExecC2C(plan, device_idata, device_idata, CUFFT_FORWARD);
//            cudaThreadSynchronize();
//            cudaEventRecord(event_stop, 0);
//            cudaEventSynchronize(event_stop);
//            cudaEventElapsedTime(&timestamp, event_start, event_stop);
//            checkCUDAError("Performing FFT");
//            printf("%d: Performed forward FFT: %lf\n",  (int) (time(NULL) - start), timestamp);

            // --------------------- Coherent Dedispersion -------------------------------
//	        dim3 gridDim(fftsize / blocksize, nchans);
//                
//	        cudaEventRecord(event_start, 0);
//            coherent_dedisp<<<gridDim, blocksize >>> (device_idata, obs -> cfreq, obs -> bw, 
//                                                      obs -> dm, nchans, fftsize * numBlocks, fftsize);

//            cudaThreadSynchronize();
//            cudaEventRecord(event_stop, 0);
//            cudaEventSynchronize(event_stop);
//            cudaEventElapsedTime(&timestamp, event_start, event_stop);
//            checkCUDAError("Performing coherent dedispersion");
//            printf("%d: Performed coherent dedispersion: %lf\n",  (int) (time(NULL) - start), timestamp);

            // --------------------- IFFT channels and DMs in place -----------------------
//            cudaEventRecord(event_start, 0);
//            cufftExecC2C(plan, device_idata, device_idata, CUFFT_INVERSE);
//            cudaThreadSynchronize();
//            cudaEventRecord(event_stop, 0);
//            cudaEventSynchronize(event_stop);
//            cudaEventElapsedTime(&timestamp, event_start, event_stop);
//            checkCUDAError("Performing IFFT");
//            printf("%d: Performed inverse FFT: %lf\n",  (int) (time(NULL) - start), timestamp);

            if (obs -> folding)
            {
                // --------------------- Constant-Period folding & power -----------------------
                cudaEventRecord(event_start, 0);

                unsigned shift = fmod(currTime, obs -> period) / obs -> tsamp;
//                unsigned shift = ((int) (obs -> timestamp / obs -> tsamp)) % obs -> profile_bins;

//                if (obs -> profile_bins > obs -> nsamp)
//                {
                    dim3 gridDim( fftsize / blocksize, obs -> nchans);
	                detect_fold<<<gridDim, blocksize>>>(device_idata, device_profile, obs -> nchans, 
                                                        obs -> fftsize, obs -> numBlocks, obs -> overlap / 2,
                                                        obs -> tsamp, obs -> profile_bins, shift);
//                }
//                else
//                {
//                    printf("Using new folding kernel\n");
//                    dim3 gridDim(obs -> profile_bins / 256 + 1, nchans);
//                    detect_smallfold<<<gridDim, 256>>>(device_idata, device_profile, obs -> nchans, 
//                                                       obs -> fftsize, obs -> numBlocks, obs -> overlap / 2,
//                                                       obs -> tsamp, obs -> profile_bins, shift);
//                }

                cudaThreadSynchronize();
                cudaEventRecord(event_stop, 0);
                cudaEventSynchronize(event_stop);
                cudaEventElapsedTime(&timestamp, event_start, event_stop);
                checkCUDAError("Detection and Folding");
                printf("%d: Performed detection and folding: %lf\n", (int) (time(NULL) - start), 
                                                                     timestamp);

//            sleep(2);
//                // If we have enough profiles in each channel, create bw-wide profile
//                if (1)
//                {
//                    cudaMemset(device_tempFoldData, 0, params -> profile_size * 2);

//                    // --------------------- FFT all channels into  ------------------
//                    cudaEventRecord(event_start, 0);
//                    cufftExecR2C(profile_fplan, (cufftReal *) device_idata, device_tempFoldData);
//                    cudaThreadSynchronize();
//                    cudaEventRecord(event_stop, 0);
//                    cudaEventSynchronize(event_stop);
//                    cudaEventElapsedTime(&timestamp, event_start, event_stop);
//                    checkCUDAError("Performing profile FFT");
//                    printf("%d: Performed profile FFT: %lf\n",  (int) (time(NULL) - start), timestamp);

//                    // ------------- Vector multiply with shift components  ------------
//                    //TODO: Calculate proper value
//	                dim3 gridDim(obs -> profile_bins / (blocksize * 16), nchans);

//                    shift_channels<<<gridDim, blocksize>>>(device_tempFoldData, obs -> cfreq, 
//                                     obs -> bw, obs -> dm, obs -> profile_bins, obs -> tsamp);

//                    dim3 sumDim(obs -> profile_bins / blocksize, 1);
//                    sum_channels<<< sumDim, blocksize >>>(device_tempFoldData, obs -> profile_bins, 
//                                                          nchans);
//                    cudaThreadSynchronize();
//                    cudaEventRecord(event_stop, 0);
//                    cudaEventSynchronize(event_stop);
//                    cudaEventElapsedTime(&timestamp, event_start, event_stop);
//                    checkCUDAError("Shifted and summed channels");
//                    printf("%d: Shifted and summed channels: %lf\n",  
//                                (int) (time(NULL) - start), timestamp);

//                    // ------------- Inverse FFT channel 1 containing full profile  ------------
//                    cudaEventRecord(event_start, 0);
//                    cufftExecC2R(profile_iplan, device_tempFoldData, device_profile);
//                    cudaThreadSynchronize();
//                    cudaEventRecord(event_stop, 0);
//                    cudaEventSynchronize(event_stop);
//                    cudaEventElapsedTime(&timestamp, event_start, event_stop);
//                    checkCUDAError("Performing profile FFT");
//                    printf("%d: Performed profile FFT: %lf\n",  (int) (time(NULL) - start), timestamp);
//                }
            }

            // NOTE: overriding timestamp provided from pipeline
            currTime += (obs -> nsamp ) * obs -> tsamp;
        }

        // Wait output barrier
        ret = pthread_barrier_wait(params -> output_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
            { fprintf(stderr, "Error during barrier synchronisation 2 [thread]\n"); exit(0); }

        if(loop_counter >= params -> iterations) 
        { 
            // Collect and write output to host memory
            cudaEventRecord(event_start, 0);

            if (!obs -> folding)
                for(unsigned i = 0; i < nchans; i++)
                    for(unsigned j = 0; j < numBlocks; j++)
                        cudaMemcpy(params -> host_odata + i * nsamp + j * (fftsize - overlap),
                                   device_idata + i * numBlocks * fftsize + j * fftsize + overlap / 2,
                                   (fftsize - overlap) * sizeof(cufftComplex),
                                   cudaMemcpyDeviceToHost);
            else
                cudaMemcpy(params -> host_profile, device_profile, 
                           params -> profile_size, cudaMemcpyDeviceToHost);

            cudaEventRecord(event_stop, 0);
            cudaEventSynchronize(event_stop);
            cudaEventElapsedTime(&timestamp, event_start, event_stop);
            checkCUDAError("Copying data from GPU");
            printf("%d: Copied data from GPU %d: %f\n", (int) (time(NULL) - start), tid, timestamp);

        }

        // Acquire rw lock
        if (pthread_rwlock_rdlock(params -> rw_lock))
            { fprintf(stderr, "Unable to acquire rw_lock [thread]\n"); exit(0); }

        // Update params  
        nsamp = params -> obs -> nsamp;

        // Stopping clause
        if (((THREAD_PARAMS *) thread_params) -> stop) 
        {
            if (iters >= params -> iterations - 1) 
            {  
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

    cudaFree(device_idata);
    cudaEventDestroy(event_stop);
    cudaEventDestroy(event_start); 

    printf("%d: Exited gracefully %d\n", (int) (time(NULL) - start), tid);
    pthread_exit((void*) thread_params);
}
