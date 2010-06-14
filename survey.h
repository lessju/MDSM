#ifndef SURVEY_H_
#define SURVEY_H_

typedef struct {

    float lowdm, highdm, dmstep, sub_dmstep;
    int binsize, ndms, ncalls, calldms;

} SUBBAND_PASSES ;

typedef struct {

    // Data parameters
    float tsamp, foff, fch1;
    int nsamp, nchans, tdms;
    
    // subband dedispersion paramters
    SUBBAND_PASSES *pass_parameters;
    int num_passes, nsubs;

} SURVEY;

#endif
