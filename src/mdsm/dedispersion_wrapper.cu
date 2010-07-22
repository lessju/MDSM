#include "dedispersion_thread.cu"

extern "C" void* call_dedisperse(void* thread_params);
extern "C" DEVICES* call_initialise_devices();

extern "C" void *call_dedisperse(void* thread_params) {
    return dedisperse(thread_params);
}

extern "C" DEVICES* call_initialise_devices() {
    return initialise_devices();
}
