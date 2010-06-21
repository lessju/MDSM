#include "dedispersion_thread.cu"

extern "C" void* call_dedisperse(void* thread_params);
extern "C" DEVICE_INFO** call_initialise_devices(int *num_devices);

extern "C" void *call_dedisperse(void* thread_params) {
    return dedisperse(thread_params);
}

extern "C" DEVICE_INFO** call_initialise_devices(int *num_devices) {
    return initialise_devices(num_devices);
}
