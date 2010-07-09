#ifndef DEDISPERSION_MANAGER_H_
#define DEDISPERSION_MANAGER_H_

#include <QString>
#include "survey.h"

SURVEY *processSurveyParameters(QString filepath);
float* initialiseMDSM(SURVEY* input_survey);
int process_chunk(unsigned int data_read);
void tearDownMDSM();

#endif // DEDISPERSION_MANAGER_H_
