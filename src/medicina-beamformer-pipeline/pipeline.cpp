#include "DoubleBuffer.h"
#include "PacketChunker.h"

#include <QCoreApplication>

#include "beamforming_manager.h"
#include "survey.h"

#include <stdio.h>
#include <stdlib.h>

// Global arguments
unsigned samplesPerPacket = 128, port = 10000, bandwidth = 20e6,
         numberOfAntennas = 32,  numberOfChannels = 1024, nsamp = 4096;
        

// Process command-line parameters
void process_arguments(int argc, char *argv[])
{
    int i = 1;
    
    while((fopen(argv[i], "r")) != NULL)
        i++;
        
    if (i != 2) {
        printf("Pipeline needs observation file!\n");
        exit(-1);
    }
    
    while(i < argc) {
       if (!strcmp(argv[i], "-samplesPerPacket"))
           samplesPerPacket = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-port"))
           port = atoi(argv[++i]);
       i++;
    }
}

// Main method
int main(int argc, char *argv[])
{
    double    timestamp, sampRate;
    
    // Create mait QCoreApplication instance
    QCoreApplication app(argc, argv);

    // Process arguments
    process_arguments(argc, argv);  

    // Initialise MDSM
    SURVEY *survey = processSurveyParameters(argv[1]);
    initialise(survey);

    // Check if requested nsamp is a divisible by the incoming samples per heap
    if (survey -> nsamp % samplesPerPacket != 0)
    {
        printf("Number of samples (%d) must be exactle divisble by number of spectra per Heap (%d)\n",
                survey -> nsamp, samplesPerPacket);
        exit(0);
    } 
    
    // Initialise Circular Buffer and set thread priority
    DoubleBuffer doubleBuffer(numberOfAntennas, numberOfChannels, survey -> nsamp);
    doubleBuffer.start();
    doubleBuffer.setPriority(QThread::TimeCriticalPriority);

    // Initialise UDP Chunker and set thread priority
    PacketChunker chunker(port, numberOfAntennas, numberOfChannels, samplesPerPacket, numberOfChannels);
    chunker.setDoubleBuffer(&doubleBuffer);
    chunker.start();
    chunker.setPriority(QThread::TimeCriticalPriority);

    // Start main processing loop
    while(true) 
    {
        // Get pointer to next buffer   
        unsigned char *udpBuffer = doubleBuffer.prepareRead(&timestamp, &sampRate);

        // Get destination buffer from MDSM
        unsigned char *inputBuffer = get_buffer_pointer();

        // Copy UDP data to buffer
        memcpy(inputBuffer, udpBuffer, survey -> nantennas * survey -> nchans * 
                                       survey -> nsamp * sizeof(unsigned char));

        // Done reading from buffer
        doubleBuffer.readReady();
        
        // Call MDSM for dedispersion
        unsigned int samplesProcessed;
        next_chunk(survey -> nsamp, samplesProcessed, timestamp, sampRate);
        if (!start_processing(survey -> nsamp)) printf("MDSM stopped....\n");
    } 
}
