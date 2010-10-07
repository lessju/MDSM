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
    
    // Switch between brute-froce & subband dedisp
    bool useBruteForce;

    // Brute Force parameters
	float lowdm, dmstep;

    // subband dedispersion paramters
    SUBBAND_PASSES *pass_parameters;
    unsigned num_passes, nsubs;
    
    // Timing parameters
    long long timestamp;
    long blockRate;

    // File pointer (to be substitued with QIODevice)
    FILE *fp;

    // Number of GPUs which are used
    unsigned num_threads;
    unsigned *gpu_ids;
    unsigned num_gpus;

} SURVEY;

#endif
