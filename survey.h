#ifndef SURVEY_H_
#define SURVEY_H_

typedef struct {

    float lowdm, highdm, dmstep, sub_dmstep;
    int binsize, ndms, ncalls, calldms;

} SUBBAND_PASSES ;

typedef struct {

    // Data parameters
    int nsamp, nchans, tdms, maxshift, nbits;
    float tsamp, foff, fch1;
    
    // subband dedispersion paramters
    SUBBAND_PASSES *pass_parameters;
    int num_passes, nsubs;

    // File pointer (to be substitued with QIODevice
    FILE *fp;

} SURVEY;

#endif
