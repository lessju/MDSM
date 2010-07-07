#ifndef DEDISPERSION_MANAGER_H_
#define DEDISPERSION_MANAGER_H_

#include "survey.h"

float* initialiseMDSM(int argc, char *argv[], SURVEY* input_survey);
int process_chunk(int data_read);
void tearDownMDSM();

#endif // DEDISPERSION_MANAGER_H_
