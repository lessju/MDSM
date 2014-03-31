// MDSM stuff
#include "multibeam_dedispersion_manager.h"
#include "file_handler.h"
#include "survey.h"
#include "cache_brute_force.h"

#include <math.h>
#include <random>

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
unsigned long readBinaryData(SURVEY *survey, float *buffer, FILE *fp, int nbits, int nsamp, int nchans, int nbeams)
{
    int file_nchans = nchans * nbeams;
    float *temp_buffer = (float *) malloc(file_nchans * sizeof(float));
    size_t tot_nsamp = 0;

    // Corner turn while reading data
    // Calculate buffer mean whilst corner turning
    float sum = 0, tmpstd = 0;
    for(int i = 0; i < nsamp; i++) 
    {
        if (read_block(fp, nbits, temp_buffer, file_nchans ) != (unsigned) file_nchans)
            break;

        for(int b = 0; b < nbeams; b++)
            for(int j = 0; j < nchans; j++)
            {
                float value = temp_buffer[b * nchans + j];
                buffer[b * nchans * nsamp + j * nsamp + i] = value;
                sum += temp_buffer[b * nchans + j];
                tmpstd += pow(value, 2);
            }

        tot_nsamp++;
    }

    // Free temporary buffer
    free(temp_buffer);

    // If we haven't read all required samples, populate rest of buffer with data mean
    if (tot_nsamp == nsamp)
        return nsamp;
    else if (tot_nsamp == 0)
        return 0;

    survey -> last_nsamp = tot_nsamp;

//    float mean = sum / (float) (tot_nsamp * nbeams * nchans);
//    float std  = sqrt(tmpstd / (float) (tot_nsamp * nbeams * nchans) - mean * mean);

//    std::default_random_engine generator;
//    std::normal_distribution<double> distribution(mean, std);

//    for(unsigned i = 0; i < nbeams; i++)
//        for(unsigned j = 0; j < nchans; j++)
//            for(unsigned k = tot_nsamp; k < nsamp; k++)
//                buffer[i * nchans * nsamp + j * nsamp + k] = distribution(generator);

    return nsamp;
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
    unsigned int counter = 0, data_read = 0, orig_nsamp = survey -> nsamp;
    unsigned total_read = 0, i;
    float mean = 0;
    while (TRUE) 
    {
        // Get current buffer pointer
        input_buffer = get_buffer_pointer();

		// Read data from file
        if (counter > 0)
        {
		    data_read = readBinaryData(survey, input_buffer, survey -> fp, survey -> nbits, 
                                       orig_nsamp, survey -> nchans, survey -> nbeams);

            // Make sure that data_read is multiple of DEDISP_THREADS
            data_read = data_read - (data_read % DEDISP_THREADS);
        }
        else
        {
            data_read = readBinaryData(survey, input_buffer, survey -> fp, survey -> nbits, 
                                       survey -> beams[0].maxshift, survey -> nchans, survey -> nbeams);
        }

        // If we have more than one input beam, copy to all beams
//        for(unsigned i = 1; i < survey -> nbeams; i++)
//            memcpy(input_buffer + survey -> nsamp * survey -> nchans * i, input_buffer, 
//                                  survey -> nsamp * survey -> nchans * sizeof(float));

        // Check if there is more processing to be done
		unsigned int x;
        next_chunk_multibeam(data_read, x, survey -> tsamp * total_read, survey -> tsamp);
		if (!start_processing(data_read))  break;
       
        counter++;

        if (counter > 0)
            total_read += data_read;   
    }

    // Tear down MDSM
    tearDownMDSM();
}
