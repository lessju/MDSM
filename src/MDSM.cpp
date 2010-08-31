// MDSM stuff
#include "dedispersion_manager.h"
#include "file_handler.h"
#include "survey.h"

// QT Stuff
#include <QCoreApplication>

// C++ stuff
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <DedispersedSeries.h>
#include "DedispersedTimeSeries.h"

#define USING_PELICAN_LOFAR 0

// Process command-line parameters (when reading from file)
SURVEY* file_process_arguments(int argc, char *argv[])
{
    FILE_HEADER* header;
    SURVEY* survey;
    int i = 1, file = 0;

    if (argc < 3){
        fprintf(stderr, "Need at least data and observation files!\n");
        exit(0);
    }
    
    // Read in file to be processed from argument list
    if ((fopen(argv[i], "rb")) != NULL) 
        file = i;
    else {
        fprintf(stderr, "Invalid data file!\n");
        exit(0);
    }
      
    i++;

    // Load observation file and generate survey 
    if (!strcmp(argv[i], "-obs"))
           survey = processSurveyParameters(QString(argv[++i])); 
    else {
        fprintf(stderr, "Second argument must be observation file! [-obs filepath] \n");
        exit(0);
    }

    i++;
    
    // Set file to be processed    
    survey -> fp = fopen(argv[file], "rb");
    header = read_header(survey -> fp);
    survey -> nbits = header -> nbits;

    // Load in additional parameters
    while(i < argc) {  
       if (!strcmp(argv[i], "-nsamp"))
           survey -> nsamp = atoi(argv[++i]);  
       else { printf("Invalid parameter\n"); exit(0); }
       i++;
    }

    return survey;
}

// Process command-line parameters (when receiving data from lofar)
SURVEY* lofar_process_arguments(int argc, char *argv[])
{
    SURVEY* survey;
    int i = 1;

    if (argc < 2){
        fprintf(stderr, "Need at least observation file!\n");
        exit(0);
    }

   // Load observation file and generate survey 
    if (!strcmp(argv[i], "-obs"))
           survey = processSurveyParameters(QString(argv[++i])); 
    else {
        fprintf(stderr, "First argument must be observation file! [-survey filepath]\n");
        exit(0);
    }

    while(i < argc) {  
       if (!strcmp(argv[i], "-nsamp"))
           survey -> nsamp = atoi(argv[++i]);  
       else if (!strcmp(argv[i], "-nsubs"))
           survey -> nsubs = atoi(argv[++i]); 
       else if (!strcmp(argv[i], "-survey")) // Survey file, process
           survey = processSurveyParameters(QString(argv[++i])); 
       else { printf("Invalid parameter\n"); exit(0); }
       i++;
    }

    return survey;
}

// Load data from binary file
unsigned long readBinaryData(float *buffer, FILE *fp, int nbits, int nsamp, int nchans)
{  
    return read_block(fp, nbits, buffer, (unsigned long) nsamp * nchans) / nchans;
}

// MDSM entry point
int main(int argc, char *argv[])
{
    // Create mait QCoreApplication instance
    QCoreApplication app(argc, argv);
    SURVEY* survey = file_process_arguments(argc, argv);

    // Initialise Dedispersion code
    // NOTE: survey will be updated with MDSM parameters
    float *input_buffer = NULL;
    input_buffer = initialiseMDSM(survey);

    // Process current chunk
    unsigned int counter = 0, data_read = 0, total = 0;
    while (TRUE) {

		// READING DATA FROM FILE
		if (counter == 0) {   // First read, read in maxshift (TODO: need to be changed to handle maxshift internally

			data_read = readBinaryData(input_buffer, survey -> fp, survey -> nbits, survey -> nsamp + survey -> maxshift,
									   survey -> nchans);
			if (data_read < survey -> maxshift) {
				fprintf(stderr, "Not enough samples in file to perform dediseprsion\n");
				data_read = 0;
			}
			else
				data_read -= survey -> maxshift;
		}
		else                 // Read in normally
			data_read = readBinaryData(input_buffer, survey -> fp, survey -> nbits, survey -> nsamp, survey -> nchans);

        // Check if there is more processing to be done

		unsigned int x;
		next_chunk(data_read, x);
		if (!start_processing(data_read))  break;
       
        total += data_read;
        counter++;
    }

    // Tear down MDSM
    tearDownMDSM();
}
