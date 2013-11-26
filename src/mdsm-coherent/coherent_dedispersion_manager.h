    #ifndef COHERENT_DEDISPERSION_MANAGER_H_
#define COHERENT_DEDISPERSION_MANAGER_H_

#include "coherent_dedispersion_output.h"
#include "coherent_dedispersion_thread.h"
#include "observation.h"
#include <QString>

typedef struct
{
    // Global parameters
    OBSERVATION *obs;

    // Parameters extracted from main
    time_t start, begin;

    THREAD_PARAMS* threads_params;
    OUTPUT_PARAMS output_params;

    DEVICES* devices;
    unsigned num_devices;

    cufftComplex* host_idata, *host_odata;
    float *host_profile;

    unsigned i, ndms;
    int loop_counter;
    unsigned pnsamp, ppnsamp;
    bool outSwitch;
    pthread_rwlock_t rw_lock;
    pthread_barrier_t input_barrier, output_barrier;
    pthread_attr_t thread_attr;
    pthread_t output_thread;
    pthread_t* threads;

} MANAGER;

OBSERVATION *processObservationParameters(QString filepath);
float  *initialiseMDSM(OBSERVATION* input_obs);
void   next_coherent_chunk(unsigned int data_read, unsigned &samples, 
                           double timestamp = 0, double blockRate = 0);
int    start_coherent_processing(unsigned int data_read);
void   tearDownCoherentMDSM();

#endif // COHERENT_DEDISPERSION_MANAGER_H_
