#include "coherent_dedispersion_manager.h"
#include "unistd.h"

// QT stuff
#include <QFile>
#include <QStringList>
#include <QDomElement>

// Forward declarations
extern "C" void* call_dedisperse(void* thread_params);
extern "C" DEVICES* call_initialise_devices(OBSERVATION *obs);

#include <iostream>

MANAGER manager;

// ================================== C++ Stuff =================================
// Process observation parameters
OBSERVATION* processObservationParameters(QString filepath)
{
    QDomDocument document("observation");

    QFile file(filepath);
    if (!file.open(QFile::ReadOnly | QFile::Text))
        throw QString("Cannot open observation file '%1'").arg(filepath);

    // Read the XML configuration file into the QDomDocument.
    QString error;
    int line, column;
    if (!document.setContent(&file, true, &error, &line, &column)) {
        throw QString("Config::read(): Parse error "
                "(Line: %1 Col: %2): %3.").arg(line).arg(column).arg(error);
    }

    QDomElement root = document.documentElement();
    if( root.tagName() != "observation" )
        throw QString("Invalid root elemenent observation parameter xml file, should be 'observation'");

    // Get the root element of the observation file
    QDomNode n = root.firstChild();

    // Initalise obs object
    manager.obs = (OBSERVATION *) malloc(sizeof(OBSERVATION));

    // Assign default values
    manager.obs -> nsamp = 0;
    manager.obs -> gpu_ids = NULL;
    manager.obs -> num_gpus = 0;

    // Start parsing observation file and generate obs parameters
    n = root.firstChild();
    while( !n.isNull() )
    {
        QDomElement e = n.toElement();
        if( !e.isNull() )
        {
            if (QString::compare(e.tagName(), QString("frequencies"), Qt::CaseInsensitive) == 0) {
                manager.obs -> cfreq = e.attribute("center").toFloat();
                manager.obs -> bw = e.attribute("bandwidth").toFloat();
            }
            else if (QString::compare(e.tagName(), QString("dm"), Qt::CaseInsensitive) == 0) {
                manager.obs -> dm = e.attribute("dm").toFloat();
            }
            else if (QString::compare(e.tagName(), QString("channels"), Qt::CaseInsensitive) == 0) {
                manager.obs -> nchans = e.attribute("number").toUInt();
            }
            else if (QString::compare(e.tagName(), QString("timing"), Qt::CaseInsensitive) == 0)
            {
                manager.obs -> tsamp = e.attribute("tsamp").toFloat();
                manager.obs -> period = e.attribute("period").toFloat();
                manager.obs -> folding = e.attribute("folding").toUInt();
            }
            else if (QString::compare(e.tagName(), QString("samples"), Qt::CaseInsensitive) == 0)
			   manager.obs -> nsamp = e.attribute("number").toUInt();
        
            // Check if user has specified GPUs to use
            else if (QString::compare(e.tagName(), QString("gpus"), Qt::CaseInsensitive) == 0) {
			    QString gpus = e.attribute("ids");
                QStringList gpuList = gpus.split(",", QString::SkipEmptyParts);
                manager.obs -> gpu_ids = (unsigned *) malloc(sizeof(unsigned) * gpuList.count());
                std::cout << "GPUs to use: ";
                for(int i = 0; i < gpuList.count(); i++) {
                    (manager.obs -> gpu_ids)[i] = gpuList[i].toUInt();
                    std::cout << gpuList[i].toUInt() << " ";
                }
                manager.obs -> num_gpus = gpuList.count();
                std::cout << std::endl;
            }
        }
        n = n.nextSibling();
    }

    return manager.obs;
}

// ==================================  C Stuff ==================================

// Initliase MDSM parameters, return poiinter to input buffer where
// input data will be stored
float* initialiseMDSM(OBSERVATION* input_obs)
{
    unsigned k;
    int ret;

    // Initialise manager
    manager.start = time(NULL);
    manager.loop_counter = 0;
    manager.outSwitch = true;
    manager.rw_lock = PTHREAD_RWLOCK_INITIALIZER;

    size_t host_isize, host_osize, device_isize, device_osize, profile_size;

    // Initialise obs
    OBSERVATION *obs = input_obs;
    manager.obs = input_obs;
    // Initialise devices/thread-related variables
    pthread_attr_init(&manager.thread_attr);
    pthread_attr_setdetachstate(&manager.thread_attr, PTHREAD_CREATE_JOINABLE);

    manager.devices                = call_initialise_devices(input_obs);
    manager.num_devices            = manager.devices -> num_devices;
    manager.obs -> num_threads     = manager.num_devices;
    manager.threads                = (pthread_t *) calloc(sizeof(pthread_t), manager.num_devices);
    manager.threads_params         = (THREAD_PARAMS *) malloc(manager.num_devices * sizeof(THREAD_PARAMS));

    // Calculate required chirp length, overlap size and usable fftsize for convolution
    // chirp_len will consider the lowest frequency channel
    float lofreq = obs -> cfreq - fabs(obs -> bw / 2.0);
    float hifreq = obs -> cfreq - fabs(obs -> bw / 2.0) + fabs(obs -> bw / obs -> nchans);

    unsigned chirp_len = 4.150e6 * obs -> dm * 
                         (pow(lofreq, -2) - pow(hifreq, -2)) * abs(obs -> bw * 1e3);

    // Set overlap as the next power of two from the chirp len
    unsigned overlap   = pow(2, ceil(log2((float) chirp_len)));

    // Calculate an efficient fftsize (multiple simulataneous FFTs will be computed)
    // NOTE: Stolen from GUPPI dedisperse_gpu.cu

    unsigned fftsize = 16 * 1024;
    if      (overlap <= 1024)    fftsize = 32 * 1024;
    else if (overlap <= 2048)    fftsize = 64 * 1024;
    else if (overlap <= 16*1024) fftsize = 128 * 1024;
    else if (overlap <= 64*1024) fftsize = 256 * 1024;

    while (fftsize < 2 * overlap) fftsize *= 2;
    
    if (fftsize > 2 * 1024 * 1024) {
        printf("FFT length is too large, cannot dedisperse\n");
        exit(0);
    }

    // Calculate number of gpu samples which can be processed in the GPU, and calculate
    // number of input samples required for this (excluding the buffer edge wings)
    // We define numBlocks... for now
    // TODO: Optimise all of this
//    fftsize /= 4;
    obs -> numBlocks    = 1;
    obs -> gpuSamples   = obs -> numBlocks * (fftsize - overlap) + overlap;
    obs -> nsamp        = obs -> gpuSamples - overlap;
    obs -> fftsize      = fftsize;
    obs -> overlap      = overlap;
    obs -> wingLen      = overlap;
    obs -> profile_bins = obs -> period / obs -> tsamp;
 //    obs -> profile_bins = 1024;

    // Calculate memory sizes
    host_isize    = obs -> gpuSamples * obs -> nchans * sizeof(cufftComplex);
    host_osize    = obs -> nsamp * obs -> nchans * sizeof(cufftComplex); 
    device_isize  = obs -> numBlocks * obs -> fftsize * obs -> nchans * sizeof(cufftComplex);
    device_osize  = host_osize;
    profile_size  = obs -> profile_bins * obs -> nchans * sizeof(float);

    // Initialise buffers and create output buffer (a separate buffer for each GPU output)
    manager.host_idata   = (cufftComplex *) malloc(host_isize);
    manager.host_odata   = (cufftComplex *) malloc(manager.num_devices * host_osize);
    manager.host_profile = (float *) malloc(profile_size);

    // Log parameters    
    printf("\n\tnchans: %d, dm: %f, nsamp: %d, gpuSamples: %d\n"
           "\toverlap: %d, fftsize: %d, numBlocks: %d, profile bins: %d\n"
           "\tGPU buffer size: %.2f MB, Profile buffer size: %.2f MB\n\n",

               obs -> nchans, obs -> dm, obs -> nsamp, obs -> gpuSamples,
               overlap, fftsize, obs -> numBlocks, obs -> profile_bins,
               (fftsize * obs -> numBlocks * obs -> nchans * sizeof(cufftComplex)) / (1024 * 1024.0),
                profile_size / (1024 * 1024.0));    

    // Initialise barriers
    if (pthread_barrier_init(&manager.input_barrier, NULL, manager.num_devices + 2))
        { fprintf(stderr, "Unable to i nitialise input barrier\n"); exit(0); }

    if (pthread_barrier_init(&manager.output_barrier, NULL, manager.num_devices + 2))
        { fprintf (stderr, "Unable to initialise output barrier\n"); exit(0); }

    // Create output params and output file
    manager.output_params.nthreads        = manager.num_devices;
    manager.output_params.iterations      = 2;
    manager.output_params.maxiters        = 2;
    manager.output_params.host_odata      = manager.host_odata;
    manager.output_params.host_osize      = host_osize * manager.num_devices;
    manager.output_params.stop            = 0;
    manager.output_params.rw_lock         = &manager.rw_lock;
    manager.output_params.input_barrier   = &manager.input_barrier;
    manager.output_params.output_barrier  = &manager.output_barrier;
    manager.output_params.start           = manager.start;
    manager.output_params.obs             = obs;
    manager.output_params.host_profile    = manager.host_profile;

    // Create output thread 
    if (pthread_create(&manager.output_thread, &manager.thread_attr, 
                       process_output, (void *) &manager.output_params))
        { fprintf(stderr, "Error occured while creating output thread\n"); exit(0); }

    // Create manager.threads and assign devices
    for(k = 0; k < manager.num_devices; k++) {

        // Create THREAD_PARAMS for thread, based on input data and DEVICE_INFO
        manager.threads_params[k].iterations = 1;
        manager.threads_params[k].maxiters = 2;
        manager.threads_params[k].stop = 0;
        manager.threads_params[k].host_odata = &manager.host_odata[host_osize * k];
        manager.threads_params[k].host_idata = manager.host_idata;
        manager.threads_params[k].thread_num = k;
        manager.threads_params[k].num_threads = manager.num_devices;
        manager.threads_params[k].device_id = manager.devices -> devices[k].device_id;
        manager.threads_params[k].rw_lock = &manager.rw_lock;
        manager.threads_params[k].input_barrier = &manager.input_barrier;
        manager.threads_params[k].output_barrier = &manager.output_barrier;
        manager.threads_params[k].start = manager.start;
        manager.threads_params[k].obs = obs;
        manager.threads_params[k].host_isize = host_isize;
        manager.threads_params[k].host_osize = host_osize;
        manager.threads_params[k].device_isize = device_isize;
        manager.threads_params[k].host_osize = device_osize;
        manager.threads_params[k].profile_size = profile_size;
        manager.threads_params[k].host_profile = manager.host_profile;
        
        
         // Create thread (using function in dedispersion_thread)
         if (pthread_create(&manager.threads[k], &manager.thread_attr, 
                            call_dedisperse, (void *) &manager.threads_params[k]))
            { fprintf(stderr, "Error occured while creating thread\n"); exit(0); }
    }

    // Wait input barrier (for dedispersion_manager, first time)
    ret = pthread_barrier_wait(&manager.input_barrier);
    if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
        { fprintf(stderr, "Error during barrier synchronisation\n");  exit(0); }

    return (float *) manager.host_idata;
}

// Cleanup MDSM
void tearDownCoherentMDSM()
{
    unsigned k;

    // Join all manager.threads, making sure they had a clean cleanup
    void *status;
    for(k = 0; k < manager.num_devices; k++)
        if (pthread_join(manager.threads[k], &status))
            { fprintf(stderr, "Error while joining manager.threads\n"); exit(0); }
    pthread_join(manager.output_thread, &status);
    
    // Destroy attributes and synchronisation objects
    pthread_attr_destroy(&manager.thread_attr);
    pthread_rwlock_destroy(&manager.rw_lock);
    pthread_barrier_destroy(&manager.input_barrier);
    pthread_barrier_destroy(&manager.output_barrier);
    
    // Free memory
    free(manager.devices -> devices);
    free(manager.devices);

    free(manager.host_odata);
    free(manager.threads_params);
    free(manager.host_idata);
    free(manager.threads);

    printf("%d: Finished Process\n", (int) (time(NULL) - manager.start));
}

// Process one data chunk
void next_coherent_chunk(unsigned int data_read, unsigned &samples, double timestamp, double blockRate)
{   
    unsigned k; int ret;

    printf("%d: Read %d * 1024 samples [%d]\n", (int) (time(NULL) - manager.start),
                                                data_read / 1024, manager.loop_counter);  

    // Lock thread params through rw_lock
    if (pthread_rwlock_wrlock(&manager.rw_lock))
        { fprintf(stderr, "Error acquiring rw lock"); exit(0); }

    // Wait output barrier
    ret = pthread_barrier_wait(&manager.output_barrier);
    if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
        { fprintf(stderr, "Error during barrier synchronisation\n"); exit(0); }

    manager.ppnsamp = manager.pnsamp;
    manager.pnsamp = data_read;

    // Stopping clause (handled internally)
    if (data_read == 0) 
    { 
        manager.output_params.stop = 1;
        for(k = 0; k < manager.num_devices; k++) 
            manager.threads_params[k].stop = 1;

        // Release rw_lock
        if (pthread_rwlock_unlock(&manager.rw_lock))
            { fprintf(stderr, "Error releasing rw_lock\n"); exit(0); }

        // Return n-1 buffer
        samples = manager.ppnsamp;
    } 
    else  // Update thread params
    {
      //  Update timing parameters
      manager.obs -> timestamp = timestamp;
      manager.obs -> blockRate = blockRate;
    }

    // Release rw_lock
    if (pthread_rwlock_unlock(&manager.rw_lock))
        { fprintf(stderr, "Error releasing rw_lock\n"); exit(0); }

    if (manager.loop_counter >= 1)
    	samples = manager.ppnsamp;
    else
    	samples = -1;
}

// Start processing next chunk
int start_coherent_processing(unsigned int data_read) 
{
    int ret;

	// Stopping clause must be handled here... we need to return buffered processed data
	if (data_read == 0 && manager.outSwitch)
        manager.outSwitch = false;
	else if (data_read == 0 && !manager.outSwitch)
		return 0;

	// Wait input barrier (since input is being handled by the calling host code
    ret = pthread_barrier_wait(&manager.input_barrier);
    if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
        { fprintf(stderr, "Error during barrier synchronisation\n"); exit(0); }

    return ++manager.loop_counter;
}
