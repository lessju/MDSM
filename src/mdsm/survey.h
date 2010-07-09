#ifndef SURVEY_H_
#define SURVEY_H_

#include "stdio.h"

typedef struct {

    float lowdm, highdm, dmstep, sub_dmstep;
    int binsize, ndms, ncalls, calldms, mean, stddev;

} SUBBAND_PASSES ;

typedef struct {

    // Data parameters
    unsigned int nsamp, nchans, tdms, maxshift, nbits;
    float tsamp, foff, fch1;
    
    // subband dedispersion paramters
    SUBBAND_PASSES *pass_parameters;
    unsigned num_passes, nsubs;

    // File pointer (to be substitued with QIODevice)
    FILE *fp;

    // Number of GPUs which
    unsigned num_threads;

} SURVEY;

#endif
