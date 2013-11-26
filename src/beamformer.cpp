// MDSM stuff
#include "beamforming_manager.h"
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
    SURVEY* survey;
    int i = 1;

    if (argc < 2){
        fprintf(stderr, "Need at observation file!\n");
        exit(0);
    }

    // Load observation file and generate survey 
    if (!strcmp(argv[i], "-obs"))
           survey = processSurveyParameters(QString(argv[++i])); 
    else {
        fprintf(stderr, "Second argument must be observation file! [-obs filepath] \n");
        exit(0);
    }

    return survey;
}


// MDSM entry point
int main(int argc, char *argv[])
{
    // Create main QCoreApplication instance
    QCoreApplication app(argc, argv);
    SURVEY* survey = file_process_arguments(argc, argv);

    // Initialise Dedispersion code and survey struct
    initialise(survey);

    unsigned char *input_buffer;

    // Process current chunk
    unsigned int counter = 0, data_read = 0;
    while (TRUE) 
    {
        // Get current buffer pointer
        input_buffer = get_buffer_pointer();

		// Read data from file
//		data_read = readBinaryData(input_buffer, survey -> fp, survey -> nbits, 
//                                   survey -> nsamp, survey -> nchans, survey -> nbeams);

        // Check if there is more processing to be done
		unsigned int x;
        next_chunk(survey -> nsamp, x, survey -> tsamp * survey -> nsamp * counter, survey -> tsamp);
		if (!start_processing(survey -> nsamp))  break;
       
        counter++;
    }

    // Tear down MDSM
    tearDown();
}
