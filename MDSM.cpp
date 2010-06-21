#include "file_handler.cpp"
#include "TCPStreamer.h"
#include "survey.h"
#include <stdio.h>
#include <stdlib.h>

// Forward declarations
extern "C" float *initialiseMDSM(int argc, char *argv[], SURVEY *survey);
extern "C" void tearDownMDSM();
extern "C" int process_chunk(int data_amount);

#define TRUE 1

// To make configurable
SURVEY *processSurveyParameters()
{
    // Hard code survey parameters, for now
    SURVEY *survey = (SURVEY *) malloc(sizeof(SURVEY));

    survey -> num_passes = 3;
    survey -> pass_parameters = (SUBBAND_PASSES *) malloc(3 * sizeof(SUBBAND_PASSES)) ;
    survey -> tdms = 2112 + 792 + 484;
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
    survey -> pass_parameters[1].highdm     = 36.96;
    survey -> pass_parameters[1].dmstep     = 0.02;
    survey -> pass_parameters[1].sub_dmstep = 0.48;
    survey -> pass_parameters[1].binsize    = 2;
    survey -> pass_parameters[1].ndms       = 792;
    survey -> pass_parameters[1].calldms    = 24;
    survey -> pass_parameters[1].ncalls     = 33;

    survey -> pass_parameters[2].lowdm      = 36.96;
    survey -> pass_parameters[2].highdm     = 65.56;
    survey -> pass_parameters[2].dmstep     = 0.05;
    survey -> pass_parameters[2].sub_dmstep = 1.10;
    survey -> pass_parameters[2].binsize    = 4;
    survey -> pass_parameters[2].ndms       = 484;
    survey -> pass_parameters[2].calldms    = 22;
    survey -> pass_parameters[2].ncalls     = 22; 

    return survey;
}

// Process command-line parameters
void process_arguments(int argc, char *argv[], SURVEY* survey)
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

// Load data from Pelican-Lofar
int readTCPData(float *buffer, int nsamp)
{
}

// Load data from binary file
int readBinaryData(float *buffer, FILE *fp, int nbits, int nsamp, int nchans)
{
    return read_block(fp, nbits, buffer, nsamp * nchans) / nchans;
}

// MDSM entry point
int main(int argc, char *argv[])
{
    SURVEY *survey = processSurveyParameters();
    process_arguments(argc, argv, survey);

    // Initialise Dedispersion code
    // NOTE: survey will be updated with MDSM parameters
    float *input_buffer = NULL;
    input_buffer = initialiseMDSM(argc, argv, survey);

    // Process current chunk
    int counter = 0, data_read = 0;
    while (TRUE) {
        if (counter == 0)    // First read, read in maxshift (need to be changed to handle maxshift internally
            data_read = readBinaryData(input_buffer, survey -> fp, survey -> nbits, survey -> nsamp + survey -> maxshift, 
                                       survey -> nchans) - survey -> maxshift;
        else                 // Read in normally
            data_read = readBinaryData(input_buffer, survey -> fp, survey -> nbits, survey -> nsamp, survey -> nchans);

        if (!process_chunk(data_read)) break;

        counter++;
    }

    // Tear down MDSM
    tearDownMDSM();
}
