#include "beamforming_output.h"
#include "beamforming_thread.h"
#include "beamshifts.h"
#include "unistd.h"
#include "stdlib.h"
#include "math.h"

// QT stuff
#include <QFile>
#include <QStringList>
#include <QDomElement>

// C++ stuff
#include <iostream>

// Function macros
#define max(X, Y)  ((X) > (Y) ? (X) : (Y))

// Forward declarations
extern "C" void* call_run_beamformer(void* thread_params);
extern "C" DEVICES* call_initialise_devices(SURVEY *survey);
extern "C" void call_allocateInputBuffer(float **pointer, size_t size);
extern "C" void call_allocateOutputBuffer(float **pointer, size_t size);

// Global parameters
SURVEY  *survey;  // Pointer to survey struct
DEVICES *devices; // Pointer to devices struct
Array   *array;   // Pointer to array struct

pthread_attr_t thread_attr;
pthread_t  output_thread;
pthread_t* threads;

pthread_rwlock_t rw_lock = PTHREAD_RWLOCK_INITIALIZER;
pthread_barrier_t input_barrier, output_barrier;

THREAD_PARAMS** threads_params;
OUTPUT_PARAMS* output_params;

unsigned long *inputsize, *outputsize;
unsigned char *input_buffer;
float** output_buffer;

int loop_counter = 0;
bool outSwitch = true;
unsigned pnsamp, ppnsamp;

time_t start = time(NULL), begin;

// ============================== Safe wrapper for malloc ============================
void * safeMalloc(size_t size)
{
    void *ptr = malloc(size);
    if (ptr == NULL)
    {
        fprintf(stderr, "Error allocating memory\n");
        exit(0);
    }
    return ptr;
}

// ================================== XML-File Parser =================================
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
    survey = (SURVEY *) safeMalloc(sizeof(SURVEY));

    // Count number of pass tags
    int nbeams = 0;
    while(!n.isNull()) 
    {
        if (QString::compare(n.nodeName(), QString("beams"), Qt::CaseInsensitive) == 0) 
        {
            n = n.firstChild();
            while(!n.isNull()) 
            {
                nbeams++;
                n = n.nextSibling();
            }
        }
        n = n.nextSibling();
    }
    survey -> nbeams = nbeams;
    survey -> beams = (BEAM *) safeMalloc(nbeams * sizeof(BEAM));
    nbeams = 0;

    // Assign default values
    survey -> nsamp = 0;
    survey -> downsample = 1;
    survey -> nantennas = 32;
    survey -> nbits = 0;
    survey -> gpu_ids = NULL;
    survey -> num_gpus = 0;
    strcpy(survey -> fileprefix, "output");
    strcpy(survey -> basedir, ".");
    survey -> secs_per_file = 600;
    survey -> use_pc_time = 1;
    survey -> single_file_mode = 0;

    // Start parsing observation file and generate survey parameters
    n = root.firstChild();
    while( !n.isNull() )
    {
        QDomElement e = n.toElement();
        if( !e.isNull() )
        {
            if (QString::compare(e.tagName(), QString("channels"), Qt::CaseInsensitive) == 0) {
                survey -> perform_channelisation = e.attribute("performChannelisation", "0").toUInt();
                survey -> apply_pfb = e.attribute("applyPFB", "0").toUInt();
                survey -> ntaps = e.attribute("ntaps", "32").toUInt();
                survey -> nchans   = e.attribute("nchans").toUInt();
                survey -> subchannels = e.attribute("subchannels", "1").toUInt();
                survey -> start_channel = e.attribute("startChannel","0").toUInt();
                survey -> stop_channel = e.attribute("stopChannel","1024").toUInt();

                char *temp = e.attribute("firPath", ".").toUtf8().data();
                strcpy(survey -> fir_path, temp);
            }
            else if (QString::compare(e.tagName(), QString("antennas"), Qt::CaseInsensitive) == 0) {
                survey -> nantennas    = e.attribute("number").toUInt();
            }
            else if (QString::compare(e.tagName(), QString("timing"), Qt::CaseInsensitive) == 0)
                survey -> tsamp = e.attribute("tsamp").toFloat();
            else if (QString::compare(e.tagName(), QString("samples"), Qt::CaseInsensitive) == 0) {
			   survey -> nsamp = e.attribute("nsamp").toUInt();
			   survey -> nbits = e.attribute("nbits").toUInt();
               survey -> downsample = e.attribute("downsample").toUInt(); 
            }
            else if (QString::compare(e.tagName(), QString("writer"), Qt::CaseInsensitive) == 0) 
            {
                // Accepts two naming conventions, switches between the two 
                char *temp = e.attribute("filePrefix", "default").toUtf8().data();
                if (strcmp(temp, "default") == 0)
                    temp = e.attribute("outputFilePrefix", "output").toUtf8().data();
			    strcpy(survey -> fileprefix, temp);

                temp = e.attribute("baseDirectory", ".").toUtf8().data();
                if (strcmp(temp, "default") == 0)
                    temp = e.attribute("outputBaseDirectory", ".").toUtf8().data();
                strcpy(survey -> basedir, temp);

                int val = e.attribute("secondsPerFile", "-1").toInt();
                if (val == -1)
                    survey -> secs_per_file = e.attribute("outputSecondsPerFile", "0").toUInt();            
                else
                    survey -> secs_per_file = (unsigned) val;

                val = e.attribute("usePCTime", "-1").toInt();
                if (val == -1)
                    survey -> use_pc_time = e.attribute("outputUsePCTime", "0").toUInt();
                else
                    survey -> use_pc_time = (unsigned) val;

                val = e.attribute("singleFileMode", "-1").toInt();                
                if (val == -1)
                    survey -> single_file_mode = e.attribute("outputSingleFileMode", "0").toUInt();
                else
                    survey -> single_file_mode = (unsigned) val;

                survey -> plot_beam = e.attribute("plotBeam", "0").toUInt();
                survey -> test = e.attribute("test", "0").toUInt();
            }

            // Check if user has specified GPUs to use
            else if (QString::compare(e.tagName(), QString("gpus"), Qt::CaseInsensitive) == 0) {
			    QString gpus = e.attribute("gpuIDs");
                QStringList gpuList = gpus.split(",", QString::SkipEmptyParts);
                survey -> gpu_ids = (unsigned *) safeMalloc(sizeof(unsigned) * gpuList.count());
                for(int i = 0; i < gpuList.count(); i++)
                    (survey -> gpu_ids)[i] = gpuList[i].toUInt();
                survey -> num_gpus = gpuList.count();
            }

            // Read beam parameters
            else if (QString::compare(e.tagName(), QString("beams"), Qt::CaseInsensitive) == 0)
            {
                // Process list of beams
                if (survey -> nbeams == 0)
                    continue;

                // Generate Array structure if antenna file is provided
                QString antennaFile = e.attribute("antennaFile");
                if (!antennaFile.trimmed().isEmpty())
                    try
                    { array = processArrayFile(antennaFile); }
                    catch(QString e)
                    { std::cout << e.toUtf8().constData() << std::endl; exit(0); }

                QDomNode beam = n.firstChild();
                while (!beam.isNull())
                {
                    e = beam.toElement();    
                    survey -> beams[nbeams].beam_id = e.attribute("beamId").toUInt();
                    survey -> beams[nbeams].fch1 = e.attribute("topFrequency").toFloat();
                    survey -> beams[nbeams].foff = e.attribute("frequencyOffset").toFloat();
                    survey -> beams[nbeams].dec  = e.attribute("dec").toFloat();
                    survey -> beams[nbeams].ra   = e.attribute("ra", "0").toFloat();
                    survey -> beams[nbeams].ha   = e.attribute("ha", "0").toFloat();

                    beam = beam.nextSibling();
                    nbeams++;
                }
            }
        }
        n = n.nextSibling();
    }
    survey -> num_threads = survey -> num_gpus;

    return survey;
}


// ==============================================================================================//
// Perform all required initialisation for MDSM
void initialise(SURVEY* input_survey)
{
   unsigned i, k;

    // Initialise survey
    survey = input_survey;

    // Initialise devices
    devices = call_initialise_devices(input_survey);

    printf("============================================================================\n");

    // Calculate global nsamp for all beams
    size_t inputsize, outputsize;

    // Antenna data will be copied to the output buffer, so it's size
    // should be large enough to accommodate both
    if (survey -> nantennas == 0 || survey -> nantennas != array -> numberOfAntennas())
    {
        printf("Number of antennas should not be 0, or does not match Array configuration!\n");
        exit(0);
    }

    // Get maximum required memory and assign to both input and output buffers
    int compare = survey -> nbeams * sizeof(float);
    if (survey -> perform_channelisation)
        compare *= 2;

    if (compare / survey -> num_gpus < survey -> nantennas)
        inputsize = outputsize = survey -> nantennas * survey -> nchans * survey -> nsamp * sizeof(unsigned char);
    else
        inputsize = outputsize = survey -> nchans * survey -> nsamp * compare;

    printf("Memory Required (per GPU): Input = %d MB; Output = %d MB\n", 
          (int) (inputsize / 1024.0 / 1024.0 / survey -> num_gpus), 
          (int) (outputsize / 1024.0 / 1024.0));

    // Allocate input buffer (MDSM_STAGES separate buffer to allow dumping to disk
    // during any iteration)
//    input_buffer = (unsigned char *) safeMalloc(inputsize);
    allocateBuffer((void **) &input_buffer, inputsize);

    // Allocate output buffer
    output_buffer = (float **) safeMalloc(survey -> num_threads * sizeof(float *));
    for(unsigned i = 0; i < survey -> num_threads; i++)
        output_buffer[i] = (float *) safeMalloc(survey -> nsamp * survey -> nchans * 
                                                survey -> nbeams * sizeof(float) / survey -> downsample);

    // Allocate shifts buffer
    survey -> beam_shifts = (float2 *) safeMalloc(survey -> nchans * survey -> nbeams * survey -> nantennas * sizeof(float2));

    // Log parameters
    if (survey -> perform_channelisation)
        printf("Observation Params: nsubs = %d, nchans = %d, nsamp = %d, tsamp = %f\n", 
                survey -> nchans, survey -> subchannels, survey -> nsamp, survey -> tsamp);
    else
        printf("Observation Params: nchans = %d, nsamp = %d, tsamp = %f\n", 
                survey -> nchans, survey -> nsamp, survey -> tsamp);

    printf("Beam Params [%d beams]:\n", survey -> nbeams);
    for(i = 0; i < survey -> nbeams; i++)
    {
        BEAM beam = survey -> beams[i];
        printf("    Beam %d: fch1 = %f, foff = %f, RA = %.2f, DEC = %.2f\n", beam.beam_id, beam.fch1, beam.foff, beam.ra, beam.dec);
    }

    printf("============================================================================\n");

    // Initialise processing threads
    pthread_attr_init(&thread_attr);
    pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);
    survey -> num_threads = devices -> num_devices;
    threads = (pthread_t *) calloc(sizeof(pthread_t), survey -> num_threads);

    threads_params = (THREAD_PARAMS **) safeMalloc(survey -> num_threads * sizeof(THREAD_PARAMS*));
    for(i = 0; i < survey -> num_threads; i++)
        threads_params[i] = (THREAD_PARAMS *) safeMalloc(sizeof(THREAD_PARAMS));

    output_params = (OUTPUT_PARAMS *) safeMalloc(sizeof(OUTPUT_PARAMS));

    // Initialise barriers
    if (pthread_barrier_init(&input_barrier, NULL, survey -> num_threads + 2))
        { fprintf(stderr, "Unable to initialise input barrier\n"); exit(0); }

    if (pthread_barrier_init(&output_barrier, NULL, survey -> num_threads + 2))
        { fprintf(stderr, "Unable to initialise output barrier\n"); exit(0); }

    // Create output params and output file
    output_params -> output_buffer = output_buffer;
    output_params -> nthreads = survey -> num_threads;
    output_params -> iterations = 3;
    output_params -> maxiters = 2;
    output_params -> stop = 0;
    output_params -> rw_lock = &rw_lock;
    output_params -> input_barrier = &input_barrier;
    output_params -> output_barrier = &output_barrier;
    output_params -> start = start;
    output_params -> survey = survey;
    output_params -> output_buffer = output_buffer;

    // Create output thread 
    if (pthread_create(&output_thread, &thread_attr, process_output, (void *) output_params))
        { fprintf(stderr, "Error occured while creating output thread\n"); exit(0); }

    // Create threads and assign devices
    for(k = 0; k < survey -> num_threads; k++) {

        // Create THREAD_PARAMS for thread, based on input data and DEVICE_INFO
        // Input and output buffer pointers will point to beginning of beam
        threads_params[k] -> iterations = 1;
        threads_params[k] -> maxiters = 2;
        threads_params[k] -> stop = 0;
        threads_params[k] -> output = output_buffer;
        threads_params[k] -> input = input_buffer;
        threads_params[k] -> thread_num = k;
        threads_params[k] -> num_threads = survey -> num_threads;
        threads_params[k] -> device_id = devices -> devices[k].device_id;
        threads_params[k] -> rw_lock = &rw_lock;
        threads_params[k] -> input_barrier = &input_barrier;
        threads_params[k] -> output_barrier = &output_barrier;
        threads_params[k] -> start = start;
        threads_params[k] -> survey = survey;
        threads_params[k] -> inputsize = inputsize;
        threads_params[k] -> outputsize = outputsize;

         // Create thread (using function in dedispersion_thread)
         if (pthread_create(&threads[k], &thread_attr, call_run_beamformer, (void *) threads_params[k]))
            { fprintf(stderr, "Error occured while creating thread\n"); exit(0); }
    }

    // Wait input barrier
    int ret = pthread_barrier_wait(&input_barrier);
    if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
        { fprintf(stderr, "Error during barrier synchronisation\n");  exit(0); }
}


// We have some data available, notfity MDSM to finish previous iteration
float **next_chunk(unsigned int data_read, unsigned &samples, 
                    double timestamp = 0, double blockRate = 0)
{
    printf("%d: Read %d * 1024 samples [%d]\n", (int) (time(NULL) - start), 
                                                data_read / 1024, loop_counter);  

    // Update beam shifts for next iteration
    calculate_shifts(survey, array); 

    // Lock thread params through rw_lock
    if (pthread_rwlock_wrlock(&rw_lock))
        { fprintf(stderr, "Error acquiring rw lock"); exit(0); }

    // Wait output barrier
    int ret = pthread_barrier_wait(&output_barrier);
    if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
        { fprintf(stderr, "Error during barrier synchronisation\n"); exit(0); }

    ppnsamp = pnsamp;
    pnsamp = data_read;

    // Stopping clause (handled internally)
    if (data_read == 0) 
    { 
        survey -> nsamp     = data_read;

        output_params -> stop = 1;
        for(unsigned k = 0; k < survey -> nbeams; k++) 
            threads_params[k] -> stop = 1;

        // Release rw_lock
        if (pthread_rwlock_unlock(&rw_lock))
            { fprintf(stderr, "Error releasing rw_lock\n"); exit(0); }

        // Return n-1 buffer
        samples = ppnsamp;
        return output_buffer;
    }

    //  Update timing parameters
    survey -> timestamp = timestamp;
    survey -> blockRate = blockRate;
    survey -> nsamp     = data_read;

    // Release rw_lock
    if (pthread_rwlock_unlock(&rw_lock))
        { fprintf(stderr, "Error releasing rw_lock\n"); exit(0); }

    if (loop_counter >= 1) {
    	samples = ppnsamp;
    	return output_buffer;
    }
    else {
    	samples = -1;
    	return NULL;
    }
}

// Request a buffer pointer for the next data samples
unsigned char *get_buffer_pointer()
{
    return input_buffer;
}

// Process current data buffer
int start_processing(unsigned int data_read)
{
	// Stopping clause must be handled here... we need to return buffered processed data
	if (data_read == 0 && outSwitch)
        outSwitch = false;
	else if (data_read == 0 && !outSwitch)
		return 0;

	// Wait input barrier (since input is being handled by the calling host code
    int ret = pthread_barrier_wait(&input_barrier);
    if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
        { fprintf(stderr, "Error during barrier synchronisation\n"); exit(0); }

    return ++loop_counter;
}

// Tear down and clear/close everything
void tearDown()
{
    unsigned  k;

    // Join all threads, making sure they had a clean cleanup
    void *status;
    for(k = 0; k < survey -> nbeams; k++)
        if (pthread_join(threads[k], &status))
            { fprintf(stderr, "Error while joining threads\n"); exit(0); }
    pthread_join(output_thread, &status);

    // Destroy attributes and synchronisation objects
    pthread_attr_destroy(&thread_attr);
    pthread_rwlock_destroy(&rw_lock);
    pthread_barrier_destroy(&input_barrier);
    pthread_barrier_destroy(&output_barrier);
    
    // Free memory
    free(devices -> devices);
    free(devices);

    free(threads_params);
    free(threads);

    // TODO: Clear pinned memory and beams 
    printf("%d: Finished Process\n", (int) (time(NULL) - start));
}

