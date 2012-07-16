#include "DoubleBuffer.h"
#include "UdpChunker.h"
#include "Types.h"

#include <QCoreApplication>

#include "dedispersion_manager.h"
#include "survey.h"

#include <stdio.h>
#include <stdlib.h>

#define SQR(x) (x*x)

// Global arguments
unsigned sampPerPacket = 1, subsPerPacket = 256, 
         sampSize = 16, port = 10000, nPols = 2, sampPerSecond = 78125;

bool writeToDisk = false;

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
       if (!strcmp(argv[i], "-sampPerPacket"))
           sampPerPacket = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-subsPerPacket"))
           subsPerPacket = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-sampSize"))
           sampSize = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-port"))
           port = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-nPols"))
           nPols = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-sampPerSecond"))
           sampPerSecond = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-dumpData"))
           writeToDisk = true;
       i++;
    }
}

// Main method
int main(int argc, char *argv[])
{
    int       chansPerSubband, samples, shift, memSize;
    double    timestamp, sampRate;
    float     *inputBuffer;
    SURVEY    *survey;
    FILE      *fp = NULL;
    
    // Create mait QCoreApplication instance
    QCoreApplication app(argc, argv);

    // Process arguments
    process_arguments(argc, argv);   
    
    // Initialise MDSM
    survey = processSurveyParameters(argv[1]);
    inputBuffer = initialiseMDSM(survey);
    chansPerSubband = survey -> nchans / survey -> nsubs;
    samples = survey -> nsamp * chansPerSubband;
    shift = survey -> maxshift * chansPerSubband;
    memSize = survey -> npols * survey -> nsubs * sizeof(float);

    if ((samples - shift) < 0) {
        printf("Maxshift (%d) must be smaller than nsamp (%d)!!\n", 
                survey -> maxshift, survey -> nsamp);
        exit(-1);
    }

    // If writing to disk, initialise file
    if (writeToDisk) {
        fp = fopen("diskDump.dat", "wb");
    }
    
    // Initialise Circular Buffer
    DoubleBuffer doubleBuffer(samples, survey -> nsubs, nPols);
    
    // Initialise UDP Chunker. Data is now being read
    UDPChunker chunker(port, sampPerPacket, subsPerPacket, nPols, sampPerSecond, sampSize);
    
    // Temporary store for maxshift
    float *maxshift = (float *) malloc(shift * memSize);
    
    chunker.setDoubleBuffer(&doubleBuffer);
    chunker.start();
    chunker.setPriority(QThread::TimeCriticalPriority);
    
    // ======================== Store first maxshift =======================
    // Get pointer to next buffer
    float *udpBuffer = doubleBuffer.prepareRead(&timestamp, &sampRate);
       
    // Copy first maxshift to temporary store
    memcpy(maxshift, udpBuffer + (samples - shift) * nPols * survey -> nsubs, shift * memSize);
                     
    doubleBuffer.readReady();
    // =====================================================================

    // Start main processing loop
    while(true) {
        
        // Get pointer to next buffer
        float *udpBuffer = doubleBuffer.prepareRead(&timestamp, &sampRate);

        // Copy maxshift to buffer
        memcpy(inputBuffer, maxshift, shift * memSize);
        
        // Copy UDP data to buffer
        memcpy(inputBuffer + shift * survey -> npols * survey -> nsubs, udpBuffer, samples * memSize);

        // Copy new maxshift
	    memcpy(maxshift, udpBuffer + (samples - shift) * nPols * survey -> nsubs, shift * memSize);
	                
        doubleBuffer.readReady();
        
        // Call MDSM for dedispersion
        unsigned int samplesProcessed;
        timestamp -= (survey -> maxshift * chansPerSubband) * sampRate;
	    next_chunk(survey -> nsamp, samplesProcessed, timestamp, sampRate);
	    if (!start_processing(survey -> nsamp + survey -> maxshift)) {
	        printf("MDSM stopped....\n");
	    }

        // Write to disk if required
        if (writeToDisk) {
            short *writeBuffer = reinterpret_cast<short*>(udpBuffer);
            short *data        = reinterpret_cast<short*>(inputBuffer);

            // Calculate intensities
            for (int i = 0; i < samples; i++)
                for (unsigned j = 0; j < subsPerPacket; j++) {
                    short XvalRe = writeBuffer[i*subsPerPacket*4 + j*2];
                    short XvalIm = writeBuffer[i*subsPerPacket*4 + j*2 + 1];
                    short YvalRe = writeBuffer[i*subsPerPacket*4 + subsPerPacket*2];
                    short YvalIm = writeBuffer[i*subsPerPacket*4 + subsPerPacket*2 + 1];
                    data[i * subsPerPacket + j] = (SQR(XvalRe) + SQR(XvalIm) + SQR(YvalRe) + SQR(YvalIm));
                }

            // Write to disk
            fwrite(data, sizeof(short), samples * subsPerPacket, fp);
            fflush(fp);
            printf("Dumped data to disk... \n");
        }
    } 

    if (writeToDisk) {
        fclose(fp);
    }
}
