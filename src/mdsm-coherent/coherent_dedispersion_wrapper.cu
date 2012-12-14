#include "coherent_dedispersion_thread.cu"
#include "observation.h"

extern "C" void* call_dedisperse(void* thread_params);
extern "C" DEVICES* call_initialise_devices(OBSERVATION *obs);

extern "C" void *call_dedisperse(void* thread_params) {
    return dedisperse(thread_params);
}

extern "C" DEVICES* call_initialise_devices(OBSERVATION *obs) {
    return initialise_devices(obs);
}
