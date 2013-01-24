#include "DoubleBuffer.h"
#include "PacketChunker.h"

#include <QCoreApplication>

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
        
//    if (i != 2) {
//        printf("MDSM needs observation file!\n");
//        exit(-1);
//    }
//    
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
//    float     *inputBuffer;
    
    // Create mait QCoreApplication instance
    QCoreApplication app(argc, argv);

    // Process arguments
    process_arguments(argc, argv);   
    
    // Initialise Circular Buffer and set thread priority
    DoubleBuffer doubleBuffer(32, 1024, 65536);
    doubleBuffer.start();
    doubleBuffer.setPriority(QThread::TimeCriticalPriority);

    // Initialise UDP Chunker and set thread priority
    PacketChunker chunker(port, 32, 1024, samplesPerHeap, samplesPerSecond, 1024 * 16 );
    chunker.setDoubleBuffer(&doubleBuffer);
    chunker.start();
    chunker.setPriority(QThread::TimeCriticalPriority);

    // Start main processing loop
    while(true) 
    {
        // Get pointer to next buffer   
        char *udpBuffer = doubleBuffer.prepareRead(&timestamp, &sampRate);

        // For testing purposes, we just dump to disk for further analysis
        #if 0
        FILE *fp = fopen("Test_correlator_packet_format.dat", "wb");
        fwrite(udpBuffer, 65536 * 1024 * 32, sizeof(char), fp);
        fclose();
        #endif

        // Done reading from buffer
        doubleBuffer.readReady();
    } 
}
