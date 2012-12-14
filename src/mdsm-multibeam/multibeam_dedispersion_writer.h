#ifndef WRITER_H_
#define WRITER_H_

#include "pthread.h"
#include "survey.h"
#include "unistd.h"

typedef struct 
{
    // Thread-specific info + synchronisation objects
    unsigned short stop;

    // Data buffer
    float *writer_buffer;

    // Synchronisation object for data writing
    pthread_mutex_t *writer_mutex;
    bool data_available;

    // File options
    bool create_new_file;
    char filename[256];

    // Timing
    time_t start;

    // Survey parameters
    SURVEY *survey;

} WRITER_PARAMS;

void* write_to_disk(void* writer_params);

#endif
