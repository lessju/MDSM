// MDSM stuff
#include "multibeam_dedispersion_manager.h"
#include "file_handler.h"
#include "survey.h"

// QT Stuff
#include <QCoreApplication>

// C++ stuff
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>

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
    if (header == NULL)
        survey -> nbits = 8;
    else
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

// Load data from binary file
// This will split the bands into multiple beams (first beam highest band)
unsigned long readBinaryData(float *buffer, FILE *fp, int nbits, int nsamp, int nchans, int nbeams)
{
    int file_nchans = nchans * nbeams;
    float *temp_buffer = (float *) malloc(file_nchans * sizeof(float));
    size_t tot_nsamp = 0;

    // Corner turn while reading data
    for(int i = 0; i < nsamp; i++) 
    {
        if (read_block(fp, nbits, temp_buffer, file_nchans ) != (unsigned) file_nchans)
            return tot_nsamp;

        for(int b = 0; b < nbeams; b++)
            for(int j = 0; j < nchans; j++)
                buffer[b * nchans * nsamp + j * nsamp + i] = 
                       temp_buffer[b * nchans + j];

        tot_nsamp++;
    }

    free(temp_buffer);

    return tot_nsamp;
}

// MDSM entry point
int main(int argc, char *argv[])
{
    // Create main QCoreApplication instance
    QCoreApplication app(argc, argv);
    SURVEY* survey = file_process_arguments(argc, argv);

    // Initialise Dedispersion code and survey struct
    initialiseMDSM(survey);

    float *input_buffer;

    // Process current chunk
    unsigned int counter = 0, data_read = 0;
    while (TRUE) 
    {
        // Get current buffer pointer
        input_buffer = get_buffer_pointer();

		// Read data from file
		data_read = readBinaryData(input_buffer, survey -> fp, survey -> nbits, 
                                   survey -> nsamp, survey -> nchans, survey -> nbeams);

        // Check if there is more processing to be done
		unsigned int x;
        next_chunk_multibeam(data_read, x, survey -> tsamp * survey -> nsamp * counter, survey -> tsamp);
		if (!start_processing(data_read))  break;
       
        counter++;
    }

    // Tear down MDSM
    tearDownMDSM();
}
