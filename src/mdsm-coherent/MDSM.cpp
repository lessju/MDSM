// MDSM stuff
#include "observation.h"
#include "coherent_dedispersion_manager.h"

// C++ stuff
#include <stdio.h>
#include <stdlib.h>
#include <vector>

long int seekPos = 0;
FILE *fp;

// Process command-line parameters (when reading from file)
OBSERVATION* file_process_arguments(int argc, char *argv[])
{
    OBSERVATION* survey;
    int i = 1;

    if (argc < 3)
    {
        fprintf(stderr, "Need at least data and observation files!\n");
        exit(0);
    }
    
    // Read in file to be processed from argument list
    if ((fp = fopen(argv[i], "rb")) == NULL) 
    {
        fprintf(stderr, "Invalid data file!\n");
        exit(0);
    }
      
    i++;

    // Load observation file and generate survey 
    if (!strcmp(argv[i], "-obs"))
           survey = processObservationParameters(QString(argv[++i])); 
    else 
    {
        fprintf(stderr, "Second argument must be observation file! [-obs filepath] \n");
        exit(0);
    }

    return survey;
}

// Load data from binary file
unsigned getData(float *data, unsigned nchans, unsigned nSamples, unsigned overlap)
{
    // Seek to correct position
    if (fseek(fp, seekPos, SEEK_SET) != 0)
    {
        perror("Could not seek file\n");
        exit(0);
    }
    seekPos += nSamples * nchans * 2 * sizeof(float);

    // Read data segment
    float *buffer = (float *) malloc(nSamples * nchans * 2 * sizeof(float));
    unsigned readSamp = fread(buffer, sizeof(float), nSamples * nchans * 2, fp);

    if (readSamp < nSamples * nchans * 2)
    {
        printf("Reached end of file\n");
        return 0;
    }

	// Read data whilst transposing
	for (unsigned t = 0; t < nSamples; t++)
    {
		for(unsigned c = 0; c < nchans; c++)
		{
			data[(c * (nSamples + overlap) + overlap + t) * 2]     = buffer[t * nchans * 2 + c * 2];
			data[(c * (nSamples + overlap) + overlap + t) * 2 + 1] = buffer[t * nchans * 2 + c * 2 + 1];
		}
    }

    free(buffer);

    return nSamples;
}

// MDSM entry point
int main(int argc, char *argv[])
{
    OBSERVATION* obs = file_process_arguments(argc, argv);

    // Initialise Dedispersion code
    float *input_buffer = NULL;
    input_buffer = initialiseMDSM(obs);

    // Process current chunk
    unsigned int counter = 0, data_read = 0, total = 0;
    while (TRUE) 
    {
        if (counter == 0)
    		data_read = getData(input_buffer, obs -> nchans, obs -> gpuSamples, 0);
        else
            data_read = getData(input_buffer, obs -> nchans, 
                                obs -> gpuSamples - obs -> wingLen, obs -> wingLen);

        // Check if there is more processing to be done
		unsigned int x;
		next_coherent_chunk(data_read, x, obs -> tsamp * obs -> nsamp * counter, obs -> tsamp);
		if (!start_coherent_processing(data_read))  
            break;
       
        total += data_read;
        counter++;
    }

    // Tear down MDSM
    tearDownCoherentMDSM();
}
