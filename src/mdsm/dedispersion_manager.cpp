#include "dedispersion_output.h"
#include "dedispersion_thread.h"
#include "unistd.h"

// QT stuff
#include <QDomElement>
#include <QFile>

// Forward declarations
extern "C" void* call_dedisperse(void* thread_params);
extern "C" DEVICES* call_initialise_devices();

// Global parameters
SURVEY *survey;

// Parameters extracted from main
unsigned i, ndms, maxshift;
time_t start = time(NULL), begin;
pthread_attr_t thread_attr;
pthread_t output_thread;
DEVICES* devices;
pthread_t* threads;
THREAD_PARAMS* threads_params;
float *dmshifts;
unsigned long *inputsize, *outputsize;
float* input_buffer;
float** output_buffer;
pthread_rwlock_t rw_lock = PTHREAD_RWLOCK_INITIALIZER;
pthread_barrier_t input_barrier, output_barrier;
OUTPUT_PARAMS output_params;
int loop_counter = 0, num_devices, ret;

#include <iostream>

// ================================== C++ Stuff =================================
// Process observation parameters
SURVEY* processSurveyParameters(QString filepath)
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

    // Initalise survey object
    survey = (SURVEY *) malloc(sizeof(SURVEY));

    // Count number of pass tags
    int passes = 0;
    while(!n.isNull()) {
        if (QString::compare(n.nodeName(), QString("passes"), Qt::CaseInsensitive) == 0) {
            n = n.firstChild();
            while(!n.isNull()) {
                passes++;
                n = n.nextSibling();
            }
        }
        n = n.nextSibling();
    }

    survey -> num_passes = passes;
    survey -> pass_parameters = (SUBBAND_PASSES *) malloc(passes * sizeof(SUBBAND_PASSES));
    passes = 0;

    // Start parsing observation file and generate survey parameters
    n = root.firstChild();
    while( !n.isNull() )
    {
        QDomElement e = n.toElement();
        if( !e.isNull() )
        {
            if (QString::compare(e.tagName(), QString("frequencies"), Qt::CaseInsensitive) == 0) {
                survey -> fch1 = e.attribute("center").toFloat();
                survey -> foff = e.attribute("offset").toFloat();
            }
            else if (QString::compare(e.tagName(), QString("dm"), Qt::CaseInsensitive) == 0) ;
            else if (QString::compare(e.tagName(), QString("channels"), Qt::CaseInsensitive) == 0) {
                survey -> nchans = e.attribute("number").toUInt();
                survey -> nsubs = e.attribute("subbands").toUInt();
            }
            else if (QString::compare(e.tagName(), QString("timing"), Qt::CaseInsensitive) == 0)
                survey -> tsamp = e.attribute("tsamp").toFloat();

            // Refers to a new pass subsection
            else if (QString::compare(e.tagName(), QString("passes"), Qt::CaseInsensitive) == 0) {

                // Process list of passes
                if (survey -> num_passes == 0)
                    continue;

                QDomNode pass = n.firstChild();
                
                while (!pass.isNull()) {
                    QDomNode pass_params = pass.firstChild();

                    while (!pass_params.isNull()) {
                        e = pass_params.toElement();
                        if (QString::compare(e.tagName(), QString("lowDm"), Qt::CaseInsensitive) == 0)
                                survey -> pass_parameters[passes].lowdm = e.text().toFloat();
                        else if (QString::compare(e.tagName(), QString("highDm"), Qt::CaseInsensitive) == 0)
                                survey -> pass_parameters[passes].highdm = e.text().toFloat();            
                        else if (QString::compare(e.tagName(), QString("deltaDm"), Qt::CaseInsensitive) == 0)
                                survey -> pass_parameters[passes].dmstep = e.text().toFloat();            
                        else if (QString::compare(e.tagName(), QString("downsample"), Qt::CaseInsensitive) == 0)
                                survey -> pass_parameters[passes].binsize = e.text().toUInt();            
                        else if (QString::compare(e.tagName(), QString("subDm"), Qt::CaseInsensitive) == 0)
                                survey -> pass_parameters[passes].sub_dmstep = e.text().toFloat();            
                        else if (QString::compare(e.tagName(), QString("numDms"), Qt::CaseInsensitive) == 0)
                                survey -> pass_parameters[passes].ndms = e.text().toUInt();            
                        else if (QString::compare(e.tagName(), QString("dmsPerCall"), Qt::CaseInsensitive) == 0)
                                survey -> pass_parameters[passes].calldms = e.text().toUInt();            
                        else if (QString::compare(e.tagName(), QString("ncalls"), Qt::CaseInsensitive) == 0)
                                survey -> pass_parameters[passes].ncalls = e.text().toUInt();            

                        pass_params = pass_params.nextSibling();
                    }
                    passes++;
                    pass = pass.nextSibling();    // Go to next pass element
                }
            }
        }
        n = n.nextSibling();
    }
    survey -> tdms = 0;
    for(i = 0; i < survey -> num_passes; i++)
        survey -> tdms += survey -> pass_parameters[i].ndms;

    // TODO: Survey parameter checks
    // Assign default values;
    survey -> nsamp = 0;
    survey -> fp = NULL;
    survey -> nbits = 0;

    return survey;
}

// ==================================  C Stuff ==================================

// Return the maximum of two numbers
inline int max(int a, int b) {
  return a > b ? a : b;
}

// Calculate number of samples which can be loaded at once
int calculate_nsamp(int maxshift, size_t *inputsize, size_t* outputsize, unsigned long int memory)
{
    unsigned int i, input = 0, output = 0, chans = 0;

    for(i = 0; i < survey -> num_passes; i++) {
        input += survey -> nsubs * (survey -> pass_parameters[i].ncalls / num_devices) / survey -> pass_parameters[i].binsize;
        output += (((survey -> pass_parameters[i].ncalls / num_devices) * survey -> pass_parameters[i].calldms)) / survey -> pass_parameters[i].binsize;
        chans += survey -> nchans / survey -> pass_parameters[i].binsize;
    }

    // First pass's binsize is greater than 1, override input
    if (survey -> pass_parameters[0].binsize > 1) {
        input = survey -> nsubs * (survey -> pass_parameters[0].ncalls / num_devices);
        chans = survey -> nchans;
    }

    if (survey -> nsamp == 0)
        survey -> nsamp = ((memory * 256 * 0.95) / (max(input, chans) + max(output, input))) - maxshift;

    survey -> nsamp = 128 * 1024; // Temporary hack

    // Round down nsamp to multiple of the largest binsize
    if (survey -> nsamp % survey -> pass_parameters[survey -> num_passes - 1].binsize != 0)
        survey -> nsamp -= survey -> nsamp % survey -> pass_parameters[survey -> num_passes - 1].binsize;

    *inputsize = max(input, chans) * (survey -> nsamp + maxshift) * sizeof(float);  
    *outputsize = max(output, input) * (survey -> nsamp + maxshift) * sizeof(float);
    printf("Input size: %d MB, output size: %d MB\n", (int) (*inputsize / 1024 / 1024), (int) (*outputsize/1024/1024));

    return survey -> nsamp;
}

// DM delay calculation
float dmdelay(float f1, float f2)
{
  return(4148.741601 * ((1.0 / f1 / f1) - (1.0 / f2 / f2)));
}

// Initliase MDSM parameters, return poi/home/lessju/Code/MDSM/src/mdsm/dedispersion_manager.cpp: In function ‘float* inter to input buffer where
// input data will be stored
float* initialiseMDSM(SURVEY* input_survey)
{
    int k;

    // Initialise survey
    survey = input_survey;

    // Initialise devices/thread-related variables
    pthread_attr_init(&thread_attr);
    pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);

    devices = call_initialise_devices();
    num_devices = devices -> num_devices;
    survey -> num_threads = num_devices;
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
    survey -> nsamp = calculate_nsamp(maxshift, inputsize, outputsize, devices -> minTotalGlobalMem);

    // Initialise buffers and create output buffer (a separate buffer for each GPU output)
    input_buffer = (float *) malloc(*inputsize);
    output_buffer = (float **) malloc(num_devices * sizeof(float *));
    for(k = 0; k < num_devices; k++)
        output_buffer[k] = (float *) malloc(*outputsize);

    // Log parameters
    printf("nchans: %d, nsamp: %d, tsamp: %f, foff: %f, fch1: %f\n", survey -> nchans, 
           survey -> nsamp, survey -> tsamp, survey -> foff, survey -> fch1);
    printf("ndms: %d, max dm: %f, maxshift: %d\n", survey -> tdms, survey -> pass_parameters[survey -> num_passes - 1].highdm, maxshift);

    if (pthread_barrier_init(&input_barrier, NULL, num_devices + 2))
        { fprintf(stderr, "Unable to i nitialise input barrier\n"); exit(0); }

    if (pthread_barrier_init(&output_barrier, NULL, num_devices + 2))
        { fprintf (stderr, "Unable to initialise output barrier\n"); exit(0); }

    // Create output params and output file
    output_params.nthreads = num_devices;
    output_params.iterations = 2;
    output_params.maxiters = 2;
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
        { fprintf(stderr, "Error occured while creating output thread\n"); exit(0); }

    // Create threads and assign devices
    for(k = 0; k < num_devices; k++) {

        // Create THREAD_PARAMS for thread, based on input data and DEVICE_INFO
        threads_params[k].iterations = 1;
        threads_params[k].maxiters = 2;
        threads_params[k].stop = 0;
        threads_params[k].maxshift = maxshift;
        threads_params[k].binsize = 1;
        threads_params[k].output = output_buffer[k];
        threads_params[k].input = input_buffer;
        threads_params[k].dmshifts = dmshifts;
        threads_params[k].thread_num = k;
        threads_params[k].num_threads = num_devices;
        threads_params[k].device_id = devices -> devices[k].device_id;
        threads_params[k].rw_lock = &rw_lock;
        threads_params[k].input_barrier = &input_barrier;
        threads_params[k].output_barrier = &output_barrier;
        threads_params[k].start = start;
        threads_params[k].survey = survey;
        threads_params[k].inputsize = *inputsize;
        threads_params[k].outputsize = *outputsize;

         // Create thread (using function in dedispersion_thread)
         if (pthread_create(&threads[k], &thread_attr, call_dedisperse, (void *) &threads_params[k]))
            { fprintf(stderr, "Error occured while creating thread\n"); exit(0); }
    }

    // Wait input barrier (for dedispersion_manager, first time)
    ret = pthread_barrier_wait(&input_barrier);
    if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
        { fprintf(stderr, "Error during barrier synchronisation\n");  exit(0); }

    return input_buffer;
}

// Cleanup MDSM
void tearDownMDSM()
{
    int k;

    // Join all threads, making sure they had a clean cleanup
    void *status;
    for(k = 0; k < num_devices; k++)
        if (pthread_join(threads[k], &status))
            { fprintf(stderr, "Error while joining threads\n"); exit(0); }
    pthread_join(output_thread, &status);
    
    // Destroy attributes and synchronisation objects
    pthread_attr_destroy(&thread_attr);
    pthread_rwlock_destroy(&rw_lock);
    pthread_barrier_destroy(&input_barrier);
    pthread_barrier_destroy(&output_barrier);
    
    // Free memory
    for(k = 0; k < num_devices; k++)
       free(output_buffer[k]);

    free(devices -> devices);
    free(devices);

    free(output_buffer);
    free(threads_params);
    free(input_buffer);
    free(dmshifts);
    free(threads);

    printf("%d: Finished Process\n", (int) (time(NULL) - start));
}

// Process one data chunk
int process_chunk(unsigned int data_read, long long timestamp = 0, long blockRate = 0)
{   
    int k;

    printf("%d: Read %d * 1024 samples [%d]\n", (int) (time(NULL) - start), data_read / 1024, loop_counter);  

    // Lock thread params through rw_lock
    if (pthread_rwlock_wrlock(&rw_lock))
        { fprintf(stderr, "Error acquiring rw lock"); exit(0); }

    // Wait output barrier
    ret = pthread_barrier_wait(&output_barrier);
    if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
        { fprintf(stderr, "Error during barrier synchronisation\n"); exit(0); }

    // Stopping clause (handled internally)
    if (data_read == 0) { 
        output_params.stop = 1;
        for(k = 0; k < num_devices; k++) 
            threads_params[k].stop = 1;

        // Release rw_lock
        if (pthread_rwlock_unlock(&rw_lock))
            { fprintf(stderr, "Error releasing rw_lock\n"); exit(0); }

        // Reach barriers maxiters times to wait for rest to process
        for(i = 0; i < 2 - 1; i++) {
            pthread_barrier_wait(&input_barrier);
            pthread_barrier_wait(&output_barrier);
        }  
        return 0;

    // Update thread params
    } else {

      if (data_read < survey -> nsamp) {
          // Round down nsamp to multiple of the largest binsize
          if (data_read % survey -> pass_parameters[survey -> num_passes - 1].binsize != 0)
              data_read -= data_read % survey -> pass_parameters[survey -> num_passes - 1].binsize;

            output_params.survey -> nsamp = data_read;
            for(k = 0; k < num_devices; k++)
                threads_params[k].survey -> nsamp = data_read;
      }

      //  Update timing parameters
      survey -> timestamp = timestamp;
      survey -> blockRate = blockRate;
    }

    // Release rw_lock
    if (pthread_rwlock_unlock(&rw_lock))
        { fprintf(stderr, "Error releasing rw_lock\n"); exit(0); }

    // Wait input barrier (since input is being handled by the calling host code
    ret = pthread_barrier_wait(&input_barrier);
    if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
        { fprintf(stderr, "Error during barrier synchronisation\n"); exit(0); }

    return ++loop_counter;
}

