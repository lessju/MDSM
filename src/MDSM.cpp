// MDSM stuff
#include "dedispersion_manager.h"
#include "PelicanLofarClient.h"
#include "file_handler.h"
#include "survey.h"

// QT Stuff
#include <QCoreApplication>

// C++ stuff
#include <stdio.h>
#include <stdlib.h>

#define USING_PELICAN_LOFAR 0

// To make configurable
SURVEY *processSurveyParameters()
{
    // Hard code survey parameters, for now
    SURVEY *survey = (SURVEY *) malloc(sizeof(SURVEY));

    survey -> num_passes = 3;
    survey -> pass_parameters = (SUBBAND_PASSES *) malloc(3 * sizeof(SUBBAND_PASSES)) ;
    survey -> tdms = 2112 + 792 + 572;
    survey -> fp = NULL;

    survey -> pass_parameters[0].lowdm      = 0;
    survey -> pass_parameters[0].highdm     = 21.12;
    survey -> pass_parameters[0].dmstep     = 0.01;
    survey -> pass_parameters[0].sub_dmstep = 0.24;
    survey -> pass_parameters[0].binsize    = 1;
    survey -> pass_parameters[0].ndms       = 2112;
    survey -> pass_parameters[0].calldms    = 24;
    survey -> pass_parameters[0].ncalls     = 88;

    survey -> pass_parameters[1].lowdm      = 21.12;
    survey -> pass_parameters[1].highdm     = 36.47;
    survey -> pass_parameters[1].dmstep     = 0.02;
    survey -> pass_parameters[1].sub_dmstep = 0.48;
    survey -> pass_parameters[1].binsize    = 2;
    survey -> pass_parameters[1].ndms       = 792;
    survey -> pass_parameters[1].calldms    = 24;
    survey -> pass_parameters[1].ncalls     = 32; 

    survey -> pass_parameters[2].lowdm      = 36.47;
    survey -> pass_parameters[2].highdm     = 65.08;
    survey -> pass_parameters[2].dmstep     = 0.05;
    survey -> pass_parameters[2].sub_dmstep = 1.10;
    survey -> pass_parameters[2].binsize    = 4;
    survey -> pass_parameters[2].ndms       = 572;
    survey -> pass_parameters[2].calldms    = 26;
    survey -> pass_parameters[2].ncalls     = 20; 


    return survey;
}

// Process command-line parameters (when reading from file)
void file_process_arguments(int argc, char *argv[], SURVEY* survey)
{
    int i = 1;
    
    while((fopen(argv[i], "rb")) != NULL) {
        if (survey -> fp != NULL) {
            fprintf(stderr, "Only one file can be processed!\n");
            exit(0);
        }
        
        survey -> fp = fopen(argv[i], "rb");
        FILE_HEADER *header = read_header(survey -> fp);

        survey -> nsamp = 0;
        survey -> nchans = header -> nchans;
        survey -> tsamp = header -> tsamp;
        survey -> fch1 = header -> fch1;
        survey -> foff = header -> foff;
        survey -> nbits = header -> nbits;
        survey -> nsubs = 32;
        survey -> tdms = 0;        
        i++;
    }

    while(i < argc) {  
       if (!strcmp(argv[i], "-nsamp"))
           survey -> nsamp = atoi(argv[++i]);  
       else if (!strcmp(argv[i], "-nsubs"))
           survey -> nsubs = atoi(argv[++i]); 
       else { printf("Invalid parameter\n"); exit(0); }
       i++;
    }

    for(i = 0; i < survey -> num_passes; i++)
        survey -> tdms += survey -> pass_parameters[i].ndms;
}

// Process command-line parameters (when receiving data from lofar)
void lofar_process_arguments(int argc, char *argv[], SURVEY* survey)
{
    int i = 1;
    
    survey -> nsamp = 0;
    survey -> nsubs = 32;
    survey -> nchans = 16384;
    survey -> tsamp = 0.00264;
    survey -> fch1 = 240;
    survey -> foff = -0.000369;

    while(i < argc) {  
       if (!strcmp(argv[i], "-nsamp"))
           survey -> nsamp = atoi(argv[++i]);  
       else if (!strcmp(argv[i], "-nchans"))
           survey -> nsubs = atoi(argv[++i]); 
       else if (!strcmp(argv[i], "-tsamp"))
           survey -> nsamp = atof(argv[++i]);  
       else if (!strcmp(argv[i], "-fch1"))
           survey -> nsubs = atof(argv[++i]); 
       else if (!strcmp(argv[i], "-foff"))
           survey -> nsamp = atof(argv[++i]);  
       else if (!strcmp(argv[i], "-nsubs"))
           survey -> nsubs = atoi(argv[++i]); 
       else { printf("Invalid parameter\n"); exit(0); }
       i++;
    }

    survey -> tdms = 0;
    for(i = 0; i < survey -> num_passes; i++)
        survey -> tdms += survey -> pass_parameters[i].ndms;
}

// Load data from binary file
int readBinaryData(float *buffer, FILE *fp, int nbits, int nsamp, int nchans)
{
    return read_block(fp, nbits, buffer, nsamp * nchans) / nchans;
}

// MDSM entry point
int main(int argc, char *argv[])
{
    // Create mait QCoreApplication instance
    QCoreApplication app(argc, argv);

    // Process arguments
    SURVEY *survey = processSurveyParameters();

    #if USING_PELICAN_LOFAR == 1
        // Initialiase Pelican Lofar client if using it
        lofar_process_arguments(argc, argv, survey);
        PelicanLofarClient lofarClient("ChannelisedStreamData", "127.0.0.1", 6969);
    #else
        file_process_arguments(argc, argv, survey);
    #endif

    // Initialise Dedispersion code
    // NOTE: survey will be updated with MDSM parameters
    float *input_buffer = NULL;
    input_buffer = initialiseMDSM(argc, argv, survey);

    // Process current chunk
    int counter = 0, data_read = 0;
    while (TRUE) {

        #if USING_PELICAN_LOFAR == 1
            //  RECEIVING DATA FROM LOFAR (OR LOFAR EMULATOR)
            if (counter == 0)
                data_read = lofarClient.getNextBuffer(input_buffer, survey -> nsamp + survey -> maxshift) - survey -> maxshift;
            else
                data_read = lofarClient.getNextBuffer(input_buffer, survey -> nsamp);
        #else
            // READING DATA FROM FILE
            if (counter == 0)    // First read, read in maxshift (TODO: need to be changed to handle maxshift internally
                data_read = readBinaryData(input_buffer, survey -> fp, survey -> nbits, survey -> nsamp + survey -> maxshift, 
                                           survey -> nchans) - survey -> maxshift;
            else                 // Read in normally 
                data_read = readBinaryData(input_buffer, survey -> fp, survey -> nbits, survey -> nsamp, survey -> nchans);
        #endif
    
        if (!process_chunk(data_read)) break;

        counter++;
    }

    // Tear down MDSM
    tearDownMDSM();
}
