#include "dedispersion_output.h"
#include "dedispersion_thread.h"
#include "unistd.h"

// Forward declarations
extern "C" void* call_dedisperse(void* thread_params);
extern "C" DEVICE_INFO** call_initialise_devices(int *num_devices);

// Global parameters
SURVEY *survey;

// Parameters extracted from main
int i, ret, ndms, maxshift, num_devices;
time_t start = time(NULL), begin;
pthread_attr_t thread_attr;
pthread_t output_thread;
DEVICE_INFO** devices;
pthread_t* threads;
THREAD_PARAMS* threads_params;
float *dmshifts;
size_t *inputsize, *outputsize;
float* input_buffer;
float** output_buffer;
pthread_rwlock_t rw_lock = PTHREAD_RWLOCK_INITIALIZER;
pthread_barrier_t input_barrier, output_barrier;
OUTPUT_PARAMS output_params;
int loop_counter = 0;

static int max(int a, int b) {
  return a > b ? a : b;
}

void report_error(char* description)
{
   fprintf(stderr, description);
   exit(0);
}

// Calculate number of samples which can be loaded at once
int calculate_nsamp(int maxshift, size_t *inputsize, size_t* outputsize)
{
    unsigned int i, input = 0, output = 0, chans = 0;
    for(i = 0; i < survey -> num_passes; i++) {
        input += survey -> nsubs * survey -> pass_parameters[i].ncalls / survey -> pass_parameters[i].binsize;
        output += survey -> pass_parameters[i].ndms / survey -> pass_parameters[i].binsize;
        chans += survey -> nchans / survey -> pass_parameters[i].binsize;
    }

    if (survey -> nsamp == 0) 
        survey -> nsamp = ((1024 * 1024 * 1000) / (max(input, chans) + max(output, input))) - maxshift;

    // Round down nsamp to multiple of the largest binsize
    if (survey -> nsamp % survey -> pass_parameters[survey -> num_passes - 1].binsize != 0)
        survey -> nsamp -= survey -> nsamp % survey -> pass_parameters[survey -> num_passes - 1].binsize;

    // TODO: Correct maxshift calculation (when applied to input variable)
    *inputsize = (max(input, chans) * survey -> nsamp + maxshift * max(input, survey -> nchans)) * sizeof(float);  
    *outputsize = max(output, input) * (survey -> nsamp + maxshift) * sizeof(float);
    printf("Input size: %d MB, output size: %d MB\n", (int) (*inputsize / 1024 / 1024), (int) (*outputsize/1024/1024));

    return survey -> nsamp;
}

// DM delay calculation
float dmdelay(float f1, float f2)
{
  return(4148.741601 * ((1.0 / f1 / f1) - (1.0 / f2 / f2)));
}

// Initliase MDSM parameters, return pointer to input buffer where
// input data will be stored
float* initialiseMDSM(int argc, char *argv[], SURVEY* input_survey)
{
    // Initialise survey
    survey = input_survey;

    // Initialise devices/thread-related variables
    pthread_attr_init(&thread_attr);
    pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);

    devices = call_initialise_devices(&num_devices);
    threads = (pthread_t *) calloc(sizeof(pthread_t), num_devices);
    threads_params = (THREAD_PARAMS *) malloc(num_devices * sizeof(THREAD_PARAMS));

    // Calculate temporary DM-shifts
    dmshifts = (float *) malloc(survey -> nchans * sizeof(float));
    for (i = 0; i < survey -> nchans; i++)
          dmshifts[i] = dmdelay(survey -> fch1 + (survey -> foff * i), survey -> fch1);

    // Calculate maxshift (maximum for all threads)
    // TODO: calculate proper maxshift
    maxshift = dmshifts[survey -> nchans - 1] * survey -> pass_parameters[survey -> num_passes - 1].highdm / survey -> tsamp;  
    survey -> maxshift = maxshift;

    // Calculate nsamp
    inputsize = (size_t *) malloc(sizeof(size_t));
    outputsize = (size_t *) malloc(sizeof(size_t));
   
    survey -> nsamp = calculate_nsamp(maxshift, inputsize, outputsize);

    // Initialise buffers and create output buffer (a separate buffer for each GPU output)
    // TODO: Change to use all GPUs
    input_buffer = (float *) malloc(*inputsize);
    output_buffer = (float **) malloc(num_devices * sizeof(float *));
    for (i = 0; i < num_devices; i++)
        output_buffer[i] = (float *) malloc(*outputsize);

    // Log parameters
    printf("nchans: %d, nsamp: %d, tsamp: %f, foff: %f\n", survey -> nchans, survey -> nsamp, survey -> tsamp, survey -> foff);
    printf("ndms: %d, max dm: %f, maxshift: %d\n", survey -> tdms, survey -> pass_parameters[survey -> num_passes - 1].highdm, maxshift);

    if (pthread_barrier_init(&input_barrier, NULL, num_devices + 2))
        report_error("Unable to initialise input barrier\n");

    if (pthread_barrier_init(&output_barrier, NULL, num_devices + 2))
        report_error("Unable to initialise output barrier\n");

    // Create output params and output file
     output_params.nchans = survey -> nchans;
     output_params.nsamp = survey -> nsamp;
     output_params.nthreads = num_devices;
     output_params.iterations = 2;
     output_params.maxiters = 2;
     output_params.ndms = survey -> tdms;
     output_params.output_buffer = output_buffer;
     output_params.stop = 0;
     output_params.rw_lock = &rw_lock;
     output_params.input_barrier = &input_barrier;
     output_params.output_barrier = &output_barrier;
     output_params.start = start;
     output_params.output_file = fopen("output.dat", "w");
     output_params.survey = survey;

    // Create output thread 
    if (pthread_create(&output_thread, &thread_attr, process_output, (void *) &output_params))
        report_error("Error occured while creating output thread\n");

    // Create threads and assign devices
    for(i = 0; i < num_devices; i++) {

        // Create THREAD_PARAMS for thread, based on input data and DEVICE_INFO
        threads_params[i].iterations = 1;
        threads_params[i].maxiters = 2;
        threads_params[i].stop = 0;
        threads_params[i].nchans = survey -> nchans;
        threads_params[i].nsamp = survey -> nsamp;
        threads_params[i].tsamp = survey -> tsamp;
        threads_params[i].maxshift = maxshift;
        threads_params[i].binsize = 1;
        threads_params[i].output = output_buffer[i];
        threads_params[i].input = input_buffer;
        threads_params[i].ndms = 0;
        threads_params[i].dmshifts = dmshifts;
        threads_params[i].startdm = 0;
        threads_params[i].dmstep = 0;
        threads_params[i].thread_num = i;
        threads_params[i].device_id = devices[i] -> device_id;
        threads_params[i].rw_lock = &rw_lock;
        threads_params[i].input_barrier = &input_barrier;
        threads_params[i].output_barrier = &output_barrier;
        threads_params[i].start = start;
        threads_params[i].survey = survey;
        threads_params[i].inputsize = *inputsize;
        threads_params[i].outputsize = *outputsize;

         // Create thread (using function in dedispersion_thread)
         if (pthread_create(&threads[i], &thread_attr, call_dedisperse, (void *) &threads_params[i]))
             report_error("Error occured while creating thread\n");
    }

    // Wait input barrier (for dedispersion_manager, first time)
    ret = pthread_barrier_wait(&input_barrier);
    if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
        report_error("Error during barrier synchronisation\n"); 

    return input_buffer;
}

// Cleanup MDSM
void tearDownMDSM()
{
    // Join all threads, making sure they had a clean cleanup
    void *status;
    for(i = 0; i < num_devices; i++)
        if (pthread_join(threads[i], &status))
            report_error("Error while joining threads\n");
    pthread_join(output_thread, &status);
    
    // Destroy attributes and synchronisation objects
    pthread_attr_destroy(&thread_attr);
    pthread_rwlock_destroy(&rw_lock);
    pthread_barrier_destroy(&input_barrier);
    pthread_barrier_destroy(&output_barrier);
    
    // Free memory
    for(i = 0; i < num_devices; i++) {
       free(output_buffer[i]);
       free(devices[i]);
    }

    free(output_buffer);
    free(threads_params);
    free(devices);
    free(input_buffer);
    free(dmshifts);
    free(threads);

    printf("%d: Finished Process\n", (int) (time(NULL) - start));
}

// Process one data chunk
int process_chunk(int data_read)
{   
    printf("%d: Read %d * 1024 samples [%d]\n", (int) (time(NULL) - start), data_read / 1024, loop_counter);  

    // Lock thread params through rw_lock
    if (pthread_rwlock_wrlock(&rw_lock))
        report_error("Error acquiring rw lock");

    // Wait output barrier
    ret = pthread_barrier_wait(&output_barrier);
    if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
        report_error("Error during barrier synchronisation\n");  

    // Stopping clause (handled internally)
    if (data_read == 0) { 
        output_params.stop = 1;
        for(i = 0; i < num_devices; i++) 
            threads_params[i].stop = 1;

        // Release rw_lock
        if (pthread_rwlock_unlock(&rw_lock))
            report_error("Error releasing rw_lock\n");

        // Reach barriers maxiters times to wait for rest to process
        for(i = 0; i < 2 - 1; i++) {
            pthread_barrier_wait(&input_barrier);
            pthread_barrier_wait(&output_barrier);
        }  
        return 0;

    // Update thread params
    } else if (data_read < survey -> nsamp) {

      // Round down nsamp to multiple of the largest binsize
      if (data_read % survey -> pass_parameters[survey -> num_passes - 1].binsize != 0)
          data_read -= data_read % survey -> pass_parameters[survey -> num_passes - 1].binsize;

        output_params.nsamp = data_read;
        output_params.survey -> nsamp = data_read;
        for(i = 0; i < num_devices; i++) {
            threads_params[i].nsamp = data_read;
            threads_params[i].survey -> nsamp = data_read;
        }
    }

    // Release rw_lock
    if (pthread_rwlock_unlock(&rw_lock))
        report_error("Error releasing rw_lock\n");

    // Wait input barrier (since input is being handled by the calling host code
    ret = pthread_barrier_wait(&input_barrier);
    if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
        report_error("Error during barrier synchronisation\n");    

    return ++loop_counter;
}

