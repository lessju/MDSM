#include "beamforming_thread.cu"
#include "survey.h"

extern "C" void* call_run_beamformer(void* thread_params);
extern "C" DEVICES* call_initialise_devices(SURVEY *survey);
extern "C" void call_allocateInputBuffer(float **pointer, size_t size);
extern "C" void call_allocateOutputBuffer(float **pointer, size_t size);

extern "C" void *call_run_beamformer(void* thread_params) {
    return run_beamformer(thread_params);
}

extern "C" DEVICES* call_initialise_devices(SURVEY *survey) {
    return initialise_devices(survey);
}

extern "C" void call_allocateBuffer(void **pointer, size_t size){
    return allocateBuffer(pointer, size);
}
