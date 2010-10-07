#include "dedispersion_thread.cu"
#include "survey.h"

extern "C" void* call_dedisperse(void* thread_params);
extern "C" DEVICES* call_initialise_devices(SURVEY *survey);

extern "C" void *call_dedisperse(void* thread_params) {
    return dedisperse(thread_params);
}

extern "C" DEVICES* call_initialise_devices(SURVEY *survey) {
    return initialise_devices(survey);
}
