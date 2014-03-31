#ifndef DEDISPERSION_MANAGER_H_
#define DEDISPERSION_MANAGER_H_

#include "cache_brute_force.h"
#include "survey.h"
#include <QString>

// Process input XML file
SURVEY *processSurveyParameters(QString filepath);

// Perform all required initialisation for MDSM
void  initialiseMDSM(SURVEY* input_survey);

// We have some data available, notfity MDSM to finish previous iteration
float  **next_chunk_multibeam(unsigned int data_read, unsigned &samples, 
                              double timestamp = 0, double blockRate = 0);

// Request a buffer pointer for the next data samples
float *get_buffer_pointer();

// Request a buffer pointer for the next antenna samples
unsigned char *get_antenna_pointer();

// Process current data buffer
int start_processing(unsigned int data_read);

// Tear down and clear/close everything
void   tearDownMDSM();

#endif // DEDISPERSION_MANAGER_H_
