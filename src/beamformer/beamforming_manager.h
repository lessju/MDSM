#ifndef BEAMFORMING_MANAGER_H_
#define BEAMFORMING_MANAGER_H_

#include "survey.h"
#include <QString>

// Process input XML file
SURVEY *processSurveyParameters(QString filepath);

// Perform all required initialisation for MDSM
void  initialise(SURVEY* input_survey);

// We have some data available, notfity MDSM to finish previous iteration
float  **next_chunk(unsigned int data_read, unsigned &samples, 
                    double timestamp = 0, double blockRate = 0);

// Request a buffer pointer for the next data samples
unsigned char *get_buffer_pointer();

// Process current data buffer
int start_processing(unsigned int data_read);

// Tear down and clear/close everything
void   tearDown();

#endif // BEAMFORMING_MANAGER_H_
