#include "multibeam_dedispersion_output.h"
#include "multibeam_dedispersion_thread.h"
#include "multibeam_dedispersion_writer.h"
#include "unistd.h"
#include "stdlib.h"
#include "math.h"

// QT stuff
#include <QFile>
#include <QStringList>
#include <QDomElement>

#include <iostream>


// Function macros
#define max(X, Y)  ((X) > (Y) ? (X) : (Y))

// Forward declarations
extern "C" void* call_dedisperse(void* thread_params);
extern "C" DEVICES* call_initialise_devices(SURVEY *survey);
extern "C" void call_allocateInputBuffer(float **pointer, size_t size);
extern "C" void call_allocateOutputBuffer(float **pointer, size_t size);

// Global parameters
SURVEY  *survey;  // Pointer to survey struct
DEVICES *devices; // Pointer to devices struct

pthread_attr_t thread_attr;
pthread_t  output_thread;
pthread_t  writer_thread;
pthread_t* threads;

pthread_rwlock_t rw_lock = PTHREAD_RWLOCK_INITIALIZER;
pthread_barrier_t input_barrier, output_barrier;

pthread_mutex_t writer_mutex = PTHREAD_MUTEX_INITIALIZER;

THREAD_PARAMS** threads_params;
OUTPUT_PARAMS output_params;
WRITER_PARAMS writer_params;

unsigned long *inputsize, *outputsize;
unsigned char *antenna_buffer;
float** output_buffer;
float** input_buffer;
float*  writer_buffer;

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
    survey -> voltage = 0;
    survey -> nsamp = 0;
    survey -> nantennas = 32;
    survey -> nbits = 0;
    survey -> gpu_ids = NULL;
    survey -> num_gpus = 0;
    survey -> apply_beamforming = 0;
    survey -> detection_threshold = 5.0;
    survey -> ncoeffs = 8;
    survey -> apply_rfi_clipper = 0;
    survey -> spectrum_thresh = 6;
    survey -> channel_thresh = 10;
    survey -> channel_block = 1024;
    survey -> channel_mask = NULL;
    survey -> num_masks = 0;
    survey -> apply_median_filter = 0;
    survey -> apply_detrending = 0;
    strcpy(survey -> fileprefix, "output");
    strcpy(survey -> basedir, ".");
    survey -> secs_per_file = 600;
    survey -> use_pc_time = 1;
    survey -> single_file_mode = 0;
    survey -> dump_to_disk = 0;
    survey -> tbb_enabled = 0;
    survey -> apply_clustering = 0;
    survey -> apply_classification = 0;
    survey -> dbscan_min_points = 1;
    survey -> dbscan_time_range = 1;
    survey -> dbscan_dm_range = 1;
    survey -> dbscan_snr_range = 1;
    survey -> min_pulse_width = 1;
    survey -> output_bits = 32;
    survey -> output_compression = 1;

    // Start parsing observation file and generate survey parameters
    n = root.firstChild();
    while( !n.isNull() )
    {
        QDomElement e = n.toElement();
        if( !e.isNull() )
        {
            if (QString::compare(e.tagName(), QString("dm"), Qt::CaseInsensitive) == 0) {
                survey -> lowdm = e.attribute("lowDM").toFloat();
                survey -> tdms = e.attribute("numDMs").toUInt();
                survey -> dmstep = e.attribute("dmStep").toFloat();
            }
            else if (QString::compare(e.tagName(), QString("channels"), Qt::CaseInsensitive) == 0) {
                survey -> nchans   = e.attribute("nchans").toUInt();
                survey -> npols    = e.attribute("npols").toUInt();
                survey -> ncoeffs  = e.attribute("ncoeffs").toUInt();
            }
            else if (QString::compare(e.tagName(), QString("antennas"), Qt::CaseInsensitive) == 0) {
                survey -> nantennas    = e.attribute("number").toUInt();
            }
            else if (QString::compare(e.tagName(), QString("timing"), Qt::CaseInsensitive) == 0)
                survey -> tsamp = e.attribute("tsamp").toFloat();
            else if (QString::compare(e.tagName(), QString("samples"), Qt::CaseInsensitive) == 0) {
			   survey -> nsamp = e.attribute("nsamp").toUInt();
			   survey -> nbits = e.attribute("nbits").toUInt();
               survey -> voltage = e.attribute("voltage").toUInt();
            }
            else if (QString::compare(e.tagName(), QString("rfi"), Qt::CaseInsensitive) == 0) {
			   survey -> apply_rfi_clipper = e.attribute("applyRFIClipper").toUInt();
               survey -> spectrum_thresh = e.attribute("spectrumThreshold").toFloat();
               survey -> channel_thresh = e.attribute("channelThreshold").toFloat();
               survey -> channel_block = e.attribute("channelBlock").toUInt();
                
               // Process channel mask               
               QString mask = e.attribute("channelMask");
               QStringList maskList = mask.split(",", QString::SkipEmptyParts);
   
               survey -> channel_mask = (RANGE *) malloc(maskList.count() * sizeof(RANGE));
               survey -> num_masks = maskList.count();

                // For each comma-separated item, check if we have a range
                // specified as well
                for(int i = 0; i < maskList.count(); i++)
                    if (maskList[i].contains(QString("-")))
                    {
                        // We are dealing with a range, process accordingly
                        QStringList range = maskList[i].split("-", QString::SkipEmptyParts);
                        survey -> channel_mask[i].from = range[0].toUInt();
                        survey -> channel_mask[i].to = range[1].toUInt();
                    }
                    else
                        survey -> channel_mask[i].from = survey -> channel_mask[i].to = maskList[i].toUInt();
            }
            else if (QString::compare(e.tagName(), QString("detection"), Qt::CaseInsensitive) == 0) {
			   survey -> detection_threshold = e.attribute("detectionThreshold").toFloat();
               survey -> apply_median_filter = e.attribute("applyMedianFilter").toUInt();
               survey -> apply_detrending = e.attribute("applyDetrending").toUInt();
               survey -> tbb_enabled = e.attribute("enableTBB").toUInt();
            }
            else if (QString::compare(e.tagName(), QString("clustering"), Qt::CaseInsensitive) == 0) 
            {
	 		    survey -> apply_clustering  = e.attribute("applyClustering").toUInt();
                survey -> apply_classification = e.attribute("applyClassification").toUInt();  

                // Accepts two naming conventions, switches between the two 
                int tempInt = e.attribute("minPoints", "-1").toInt();
                if(tempInt == -1) 
                    survey -> dbscan_min_points = e.attribute("clusteringMinPoints").toUInt();
                else
                    survey -> dbscan_min_points = (unsigned) tempInt;

                float tempFloat =  e.attribute("timeRange", "-1").toFloat();   
                if (tempFloat == -1)
                    survey -> dbscan_time_range = e.attribute("clusteringTimeRange").toFloat();   
                else
                    survey -> dbscan_time_range = tempFloat;

                tempFloat = e.attribute("dmRange", "-1").toFloat();   
                if (tempFloat == -1)
                    survey -> dbscan_dm_range   = e.attribute("clusteringDmRange").toFloat();   
                else    
                    survey -> dbscan_dm_range = tempFloat;

                tempFloat = e.attribute("snrRange", "-1").toFloat(); 
                if (tempFloat == -1)
                    survey -> dbscan_snr_range  = e.attribute("clusteringSnrRange").toFloat(); 
                else
                    survey -> dbscan_snr_range = tempFloat;

                tempFloat = e.attribute("minPulseWidth", "-1").toFloat();
                if (tempFloat == -1)
                    survey -> min_pulse_width      = e.attribute("clusteringMinPulseWidth").toFloat();
                else
                    survey -> min_pulse_width = tempFloat;
            }
            else if (QString::compare(e.tagName(), QString("writer"), Qt::CaseInsensitive) == 0) {
                survey -> dump_to_disk = e.attribute("writeToFile").toUInt();
                survey -> output_bits = e.attribute("outputBits").toUInt();
                survey -> output_compression = e.attribute("compression").toUInt();

                // Accepts two naming conventions, switches between the two 
                char *temp = e.attribute("filePrefix", "default").toUtf8().data();
                if (strcmp(temp, "default") == 0)
                    temp = e.attribute("outputFilePrefix", "output").toUtf8().data();
                strcpy(survey -> fileprefix, temp);

                temp = e.attribute("baseDirectory", "default").toUtf8().data();
                if (strcmp(temp, "default") == 0)
                    temp = e.attribute("outputBaseDirectory", ".").toUtf8().data();
                strcpy(survey -> basedir, temp);
                printf("basedir: %s\n", temp);

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
            }
        
            // Check if user has specified GPUs to use
            else if (QString::compare(e.tagName(), QString("gpus"), Qt::CaseInsensitive) == 0) {
			    QString gpus = e.attribute("gpuIDs", "0");
                QStringList gpuList = gpus.split(",", QString::SkipEmptyParts);
                survey -> gpu_ids = (unsigned *) safeMalloc(sizeof(unsigned) * gpuList.count());
                for(int i = 0; i < gpuList.count(); i++)
                    (survey -> gpu_ids)[i] = gpuList[i].toUInt();
                survey -> num_gpus = gpuList.count();
            }

            // Read beam parameters
            else if (QString::compare(e.tagName(), QString("beams"), Qt::CaseInsensitive) == 0)
            {
                survey -> apply_beamforming = e.attribute("applyBeamforming").toUInt();

                // Process list of beams
                if (survey -> nbeams == 0)
                    continue;

                QDomNode beam = n.firstChild();
                while (!beam.isNull())
                {
                    e = beam.toElement();    
                    survey -> beams[nbeams].beam_id = e.attribute("beamId").toUInt();
                    survey -> beams[nbeams].fch1 = e.attribute("topFrequency").toFloat();
                    survey -> beams[nbeams].foff = e.attribute("frequencyOffset").toFloat();

                    beam = beam.nextSibling();
                    nbeams++;
                }
            }
        }
        n = n.nextSibling();
    }

    // Check if both dump_to_disk and tbb modes enabled, if
    // so then disbale dump_to_disk
    if (survey -> dump_to_disk && survey -> tbb_enabled)
        survey -> dump_to_disk = 0;

    return survey;
}

// ==================================  C Stuff ==================================

inline float dmdelay(float F1, float F2) 
{  return (4148.741601 * ((1.0 / F1 / F1) - (1.0 / F2 / F2))); }

// Calculate number of samples which can be loaded at once (calls appropriate method)
int calculate_nsamp(int maxshift, size_t *inputsize, size_t* outputsize, unsigned long int memory)
{
    if (survey -> nsamp == 0)
    	survey -> nsamp = ((memory * 1000 * 0.99 / sizeof(float)) - maxshift * survey -> nchans) 
                           / (survey -> nchans + survey -> tdms);

    *inputsize = (survey -> nsamp + maxshift) * survey -> nchans * sizeof(float);
    *outputsize = survey -> nsamp * survey -> tdms * sizeof(float);

	return survey -> nsamp;
}

// Initliase MDSM parameters, return poiinter to input buffer where
// input data will be stored
void initialiseMDSM(SURVEY* input_survey)
{
    unsigned i, j, k;

    // Initialise survey
    survey = input_survey;

    // Initialise devices
    devices = call_initialise_devices(input_survey);

    // Calculate temporary DM-shifts, maxshift and nsamp per beam
    unsigned greatest_maxshift = 0;
    for (i = 0; i < survey -> nbeams; i++)
    {
        // Calculate temporary shifts
        BEAM *beam = &(survey -> beams[i]);
        beam -> dm_shifts = (float *) safeMalloc(survey -> nchans * sizeof(float));
        for (j = 0; j < survey -> nchans; j++)
            beam -> dm_shifts[j] = dmdelay(beam -> fch1 + (beam -> foff * j), beam ->fch1);

        // Calculate maxshift
        float    high_dm   = survey -> lowdm + survey -> dmstep * (survey -> tdms - 1);
        unsigned maxshift  = beam -> dm_shifts[survey -> nchans - 1] * high_dm / survey -> tsamp;

        greatest_maxshift = ( maxshift > greatest_maxshift) 
                            ? maxshift : greatest_maxshift;

        // Round up maxshift to the nearest multiple of 256
        greatest_maxshift = (greatest_maxshift / 256) * 256 + 256;
    }

    // TEMPORARY: ASSIGN ALL BEAMS SAME MAXSHIFT
    for(i = 0; i < survey -> nbeams; i++)
    {
        BEAM *beam = &(survey -> beams[i]);
        beam -> maxshift = greatest_maxshift;
    }

    printf("============================================================================\n");

    // Calculate global nsamp for all beams
    size_t inputsize, outputsize;
    survey -> nsamp = calculate_nsamp(greatest_maxshift, &inputsize, &outputsize, 
                                      devices -> minTotalGlobalMem);

    // Check if nsamp is greater than maxshift
    if (greatest_maxshift > survey -> nsamp)
    {
//        fprintf(stderr, "Number of samples (%d) must be greater than maxshift (%d)\n", 
//                survey -> nsamp, greatest_maxshift);
//        exit(-1);
	printf("NOTE: Number of samples (%d) is less than maxshift (%d). This reduces performance.\n", survey -> nsamp, greatest_maxshift);
    }

    // When beamforming, antenna data will be copied to the output buffer, so it's size
    // should be large enough to accommodate both
    if (survey -> apply_beamforming)
    {
        if (survey -> nantennas == 0)
        {
            printf("Number of antennas should not be 0!\n");
            exit(0);
        }

        size_t beamforming_size = survey -> nantennas * survey -> nchans * survey -> nsamp * sizeof(unsigned char);
        outputsize = max(outputsize, beamforming_size);

         printf("Memory Required (per GPU): Input = %d MB; Output = %d MB\n", 
              (int) (survey -> nbeams * inputsize / 1024 / 1024 / survey -> num_gpus), (int) (outputsize/1024/1024));
    }
    else
    {
        printf("Memory Required (per beam): Input = %d MB; Output = %d MB\n", 
              (int) (inputsize / 1024 / 1024), (int) (outputsize/1024/1024));
    }

    // Allocate input buffer (MDSM_STAGES separate buffer to allow dumping to disk
    // during any iteration)
    input_buffer = (float **) safeMalloc(MDSM_STAGES * sizeof(float *));
//    for(i = 0; i < MDSM_STAGES; i++)
//    {
//        float *pointer;
//        allocateInputBuffer(&pointer, survey -> nbeams * survey -> nsamp * survey -> nchans * sizeof(float));
//        input_buffer[i] = pointer;
//    }


    for(i = 0; i < MDSM_STAGES; i++)
        input_buffer[i] = (float *) safeMalloc(survey -> nbeams * survey -> nsamp * 
                                               survey -> nchans * sizeof(float));

    // Allocate output buffer (one for each beam)
    output_buffer = (float **) safeMalloc(survey -> nbeams * sizeof(float *));
//    for (i=0; i < survey -> nbeams; i++)
//        output_buffer[i] = (float *) safeMalloc(survey -> nsamp * survey -> tdms * sizeof(float));

    // Allocate antenna buffer if beamforming
    antenna_buffer = NULL;
    if (survey -> apply_beamforming)
        antenna_buffer = (unsigned char *) malloc(survey -> nantennas * survey -> nchans * survey -> nsamp * sizeof(unsigned char));

    // Create writer buffer 
    // TODO: Provide kernel suggestions for high speed I/O
    writer_buffer = (float *) safeMalloc(survey -> nbeams * survey -> nsamp * 
                                         survey -> nchans * sizeof(float));

    // Log parameters
    printf("Observation Params: ndms = %d, maxDM = %f\n", survey -> tdms, 
                survey -> lowdm + survey -> dmstep * (survey -> tdms - 1));

    printf("Observation Params: nchans = %d, nsamp = %d, tsamp = %f\n", 
            survey -> nchans, survey -> nsamp, survey -> tsamp);

    printf("Beam Params:\n");
    for(i = 0; i < survey -> nbeams; i++)
    {
        BEAM beam = survey -> beams[i];
        printf("    Beam %d: fch1 = %f, foff = %f, maxshift = %d, gpuID = %d\n", 
                    beam.beam_id, beam.fch1, beam.foff, beam.maxshift, beam.gpu_id);
    }


    printf("\n");
    if (survey -> apply_beamforming) printf("- Performing beamforming\n");
    if (survey -> apply_rfi_clipper ) printf("- Clipping RFI\n");
    printf("- Performing dedispersion\n");
    if (survey -> apply_detrending ) printf("- Performing detrending\n");
    if (survey -> apply_median_filter ) printf("- Performing median filtering\n");
    if (survey -> dump_to_disk) printf("- Dump to disk mode enabled\n");
    if (survey -> apply_clustering) printf("- Applying DBSCAN\n");
    if (survey -> tbb_enabled) 	printf("- TBB mode enabled\n");
    printf("\n");

    printf("============================================================================\n");

    // Initialise processing threads
    pthread_attr_init(&thread_attr);
    pthread_attr_setdetachstate(&thread_attr, PTHREAD_CREATE_JOINABLE);
    survey -> num_threads = survey -> nbeams;
    threads = (pthread_t *) calloc(sizeof(pthread_t), survey -> num_threads);

    threads_params = (THREAD_PARAMS **) safeMalloc(survey -> num_threads * sizeof(THREAD_PARAMS*));
    for(i = 0; i < survey -> num_threads; i++)
        threads_params[i] = (THREAD_PARAMS *) safeMalloc(sizeof(THREAD_PARAMS));

    // Initialise barriers
    if (pthread_barrier_init(&input_barrier, NULL, survey -> num_threads + 2))
        { fprintf(stderr, "Unable to initialise input barrier\n"); exit(0); }

    if (pthread_barrier_init(&output_barrier, NULL, survey -> num_threads + 2))
        { fprintf(stderr, "Unable to initialise output barrier\n"); exit(0); }

    // Create GPU objects and allocate beams/threads to them
    GPU **gpus = (GPU **) malloc(survey -> num_gpus * sizeof(GPU *));
    for(i = 0; i < survey -> num_gpus; i++)
    {
        gpus[i] = (GPU *) malloc(sizeof(GPU));
        gpus[i] -> device_id   = survey -> gpu_ids[i];
        gpus[i] -> num_threads = 0;
        gpus[i] -> primary_thread = 0;
        gpus[i] -> thread_ids = (unsigned *) malloc(8 * sizeof(unsigned));        
    }

    // Allocate GPUs to beams (split beams among GPUs, one beam cannot be processed on more than 1 GPU)
    for(i = 0; i < survey -> num_threads; i++)
    {
        unsigned gpu = (survey -> gpu_ids)[i % survey -> num_gpus];

        // Update GPU properties
        for(j = 0; j < survey -> num_gpus; j++)
        {
            if (gpus[j] -> device_id == gpu)
            {
                if (gpus[j] -> num_threads == 0) gpus[j] -> primary_thread = i;
                threads_params[i] -> gpu_index = j;
                gpus[j] -> thread_ids[gpus[j] -> num_threads] = i;
                gpus[j] -> num_threads++;
                break;
            }
        }
    }  

    // Initialise GPU barriers
    for(i = 0; i < survey -> num_gpus; i++)
        if (pthread_barrier_init(&(gpus[i] -> barrier), NULL, gpus[i] -> num_threads))
            { fprintf(stderr, "Unable to initialise input barrier\n"); exit(0); }

    // Create output params and output file
    output_params.nthreads = survey -> num_threads;
    output_params.iterations = 3;
    output_params.maxiters = 2;
    output_params.output_buffer = output_buffer;
    output_params.input_buffer = input_buffer;
    output_params.stop = 0;
    output_params.rw_lock = &rw_lock;
    output_params.input_barrier = &input_barrier;
    output_params.output_barrier = &output_barrier;
    output_params.start = start;
    output_params.survey = survey;
    output_params.writer_mutex = &writer_mutex;
    output_params.writer_buffer = writer_buffer;
    output_params.writer_params = &writer_params;

    // Create output thread 
    if (pthread_create(&output_thread, &thread_attr, process_output, (void *) &output_params))
        { fprintf(stderr, "Error occured while creating output thread\n"); exit(0); }

    // Create threads and assign devices
    for(k = 0; k < survey -> num_threads; k++) {

        // Create THREAD_PARAMS for thread, based on input data and DEVICE_INFO
        // Input and output buffer pointers will point to beginning of beam
        threads_params[k] -> iterations = 2;
        threads_params[k] -> maxiters = 2;
        threads_params[k] -> stop = 0;
        threads_params[k] -> output = output_buffer;
        threads_params[k] -> input = input_buffer;
        threads_params[k] -> antenna_buffer = antenna_buffer;
        threads_params[k] -> thread_num = k;
        threads_params[k] -> num_threads = survey -> num_threads;
        threads_params[k] -> rw_lock = &rw_lock;
        threads_params[k] -> input_barrier = &input_barrier;
        threads_params[k] -> output_barrier = &output_barrier;
        threads_params[k] -> start = start;
        threads_params[k] -> survey = survey;
        threads_params[k] -> inputsize = inputsize;
        threads_params[k] -> outputsize = outputsize;
        threads_params[k] -> gpus = gpus;
        threads_params[k] -> cpu_threads = threads_params;

         // Create thread (using function in dedispersion_thread)
         if (pthread_create(&threads[k], &thread_attr, call_dedisperse, (void *) threads_params[k]))
            { fprintf(stderr, "Error occured while creating thread\n"); exit(0); }
    }

    // If we are writing to file, initialise and launch writer thread
    if (survey -> tbb_enabled || survey -> dump_to_disk)
    {
        // Initialiase and start the Data Writer thread, if required
        writer_params.survey   = survey;
        writer_params.start    = start;
        writer_params.stop     = 0;
        writer_params.writer_buffer   = writer_buffer;
        writer_params.writer_mutex    = &writer_mutex;
        writer_params.create_new_file = false;
        writer_params.data_available  = false;
        if (pthread_create(&writer_thread, &thread_attr, write_to_disk, (void *) &writer_params))
            { fprintf(stderr, "Error occured while creating thread\n"); exit(0); }
    }

    // If we're writing all the incoming data stream, it takes precedence over TBB
    if (survey -> dump_to_disk)
        survey -> tbb_enabled = false;

    // Wait input barrier (for dedispersion_manager, first time)
    int ret = pthread_barrier_wait(&input_barrier);
    if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
        { fprintf(stderr, "Error during barrier synchronisation\n");  exit(0); }
}

// Get next buffer pointer
float *get_buffer_pointer()
{
    return input_buffer[loop_counter % MDSM_STAGES];
}

// Get antenna pointer
unsigned char *get_antenna_pointer()
{
    return antenna_buffer;
}

// Cleanup MDSM
void tearDownMDSM()
{
    unsigned  k;

    // Join all threads, making sure they had a clean cleanup
    void *status;
    for(k = 0; k < survey -> nbeams; k++)
        if (pthread_join(threads[k], &status))
            { fprintf(stderr, "Error while joining threads\n"); exit(0); }
    pthread_join(output_thread, &status);
    pthread_join(writer_thread, &status);

    // Destroy attributes and synchronisation objects
    pthread_attr_destroy(&thread_attr);
    pthread_rwlock_destroy(&rw_lock);
    pthread_barrier_destroy(&input_barrier);
    pthread_barrier_destroy(&output_barrier);
    pthread_mutex_destroy(&writer_mutex);
    
    // Free memory
    free(devices -> devices);
    free(devices);

    free(threads_params);
    free(threads);

    // TODO: Clear pinned memory and beams 

    printf("%d: Finished Process\n", (int) (time(NULL) - start));
}

// Process one data chunk
float **next_chunk_multibeam(unsigned int data_read, unsigned &samples, 
                             double timestamp = 0, double blockRate = 0)
{   
    printf("%d: Read %d * 1024 samples [%d]\n", (int) (time(NULL) - start), 
                                                data_read / 1024, loop_counter);  

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

        output_params.stop = 1;
        writer_params.stop = 1;
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

// Start processing next chunk
int start_processing(unsigned int data_read) {

	// Stopping clause must be handled here... we need to return buffered processed data
	if (data_read == 0 && outSwitch)
        outSwitch = false;
	else if (data_read == 0 && !outSwitch)
		return 0;

    // We have data available, check if we want to dump to disk, and if so, do so
    if (survey -> dump_to_disk)
    {
        while (true)
        {
            pthread_mutex_unlock(&writer_mutex);
            if (!writer_params.data_available)
            {
                // Writer is idle, copy input data and start writing
                memcpy(writer_buffer, input_buffer[loop_counter % MDSM_STAGES],
                       survey -> nbeams * survey -> nsamp * survey -> nchans * sizeof(float));

                // Adjust file parameters
                writer_params.create_new_file = true;

                char pathName[256];
                strcpy(pathName, survey -> basedir);
                strcat(pathName, "/");
                strcat(pathName, survey -> fileprefix);
                strcat(pathName, "_");

                // Format timestamp 
                struct tm *tmp;
                if (survey -> use_pc_time) {
                    time_t currTime = time(NULL);
                    tmp = localtime(&currTime);                    
                }
                else {
                    time_t currTime = (time_t) survey -> timestamp;
                    tmp = localtime(&currTime);
                }       

                char tempStr[30];
                strftime(tempStr, sizeof(tempStr), "%F_%T", tmp);
                strcat(pathName, tempStr);
                strcat(pathName, "_dump_");
                sprintf(tempStr, "%d", survey -> output_bits);
                strcat(pathName, tempStr);
                strcat(pathName, "bits.dat");

                // Set filename
                memcpy(&(writer_params.filename), &pathName, 256 * sizeof(char));

                // Notify data writer
                writer_params.data_available = 1;

                // Ready, unlock mutex and return
                pthread_mutex_unlock(&writer_mutex);

                break;
            }
            usleep(10);
        }
    }

	// Wait input barrier (since input is being handled by the calling host code
    int ret = pthread_barrier_wait(&input_barrier);
    if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
        { fprintf(stderr, "Error during barrier synchronisation\n"); exit(0); }

    return ++loop_counter;
}
