#include "DoubleBuffer.h"
#include "SpeadBeamChunker.h"
#include "Types.h"

#include <QCoreApplication>

#include "multibeam_dedispersion_manager.h"
#include "survey.h"

#include <stdio.h>
#include <stdlib.h>

// Global arguments
unsigned samplesPerHeap = 128, port = 10000, samplesPerSecond = 78125, packetsPerHeap = 128;

// Process command-line parameters
void process_arguments(int argc, char *argv[])
{
    int i = 1;
    
    while((fopen(argv[i], "r")) != NULL)
        i++;
        
    if (i != 2) {
        printf("MDSM needs observation file!\n");
        exit(-1);
    }
    
    while(i < argc) {
       if (!strcmp(argv[i], "-samplesPerHeap"))
           samplesPerHeap = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-packetsPerHeap"))
           packetsPerHeap = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-port"))
           port = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-samplesPerSecond"))
           samplesPerSecond = atoi(argv[++i]);
       i++;
    }
}

// Main method
int main(int argc, char *argv[])
{
    double    timestamp, sampRate;
    float     *inputBuffer;
    SURVEY    *survey;
    
    // Create mait QCoreApplication instance
    QCoreApplication app(argc, argv);

    // Process arguments
    process_arguments(argc, argv);   
    
    // Initialise MDSM
    survey = processSurveyParameters(argv[1]);
    initialiseMDSM(survey);

    // Check if requested nsamp is a divisible by the incoming samples per heap
    if (survey -> nsamp % samplesPerHeap != 0)
    {
        printf("Number of samples (%d) must be exactle divisble by number of spectra per Heap (%d)\n",
                survey -> nsamp, samplesPerHeap);
        exit(0);
    }

    // Initialise Circular Buffer and set thread priority
    DoubleBuffer doubleBuffer(survey -> nbeams, survey -> nchans, survey -> nsamp, survey -> voltage);
    doubleBuffer.start();
    doubleBuffer.setPriority(QThread::TimeCriticalPriority);

    // Initialise UDP Chunker and set thread priority
    SpeadBeamChunker chunker(port, survey -> nbeams, survey -> nchans, samplesPerHeap, 
                             samplesPerSecond, packetsPerHeap * survey -> nbeams);

    chunker.setDoubleBuffer(&doubleBuffer);
    chunker.start();
    chunker.setPriority(QThread::TimeCriticalPriority);

    // Start main processing loop
    while(true) 
    {
        // Get pointer to next buffer
        float *udpBuffer = doubleBuffer.prepareRead(&timestamp, &sampRate);

        // Get destination buffer from MDSM
        inputBuffer = get_buffer_pointer();

        // Copy UDP data to buffer
        // TODO: This is a waste of bandwidth, need a function pointer from MDSM which will set its
        // input buffer pointer and copy directly to the GPU
        memcpy(inputBuffer, udpBuffer, survey -> nbeams * survey -> nchans * 
                                       survey -> nsamp * sizeof(float));

        // Done reading from buffer
        doubleBuffer.readReady();
        
        // Call MDSM for dedispersion
        unsigned int samplesProcessed;
        next_chunk_multibeam(survey -> nsamp, samplesProcessed, timestamp, sampRate);
       if (!start_processing(survey -> nsamp)) printf("MDSM stopped....\n");
    } 
}
