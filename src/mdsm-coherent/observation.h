#ifndef OBSERVATION_H_
#define OBSERVATION_H_

#include "stdio.h"

typedef struct {

    // Data parameters
    unsigned int nchans;
    float tsamp, cfreq, bw, dm;

    // Coherent dedispersion sample counts
    unsigned numBlocks, gpuSamples, nsamp, fftsize, overlap, wingLen;

    // Timing parameters
    double timestamp;
    double blockRate;

    // Folding parameters
    bool     folding;
    unsigned profile_bins;
    double   period;

    // Number of GPUs which are used
    unsigned num_threads;
    unsigned *gpu_ids;
    unsigned num_gpus;

} OBSERVATION;

#endif
