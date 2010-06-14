#include <stdio.h>
#include "survey.h"

// Process input parameter file to extract survey parameters
SURVEY *process_parameter_file(char *filename)
{
    /* SUBBAND TEST FILE:
    /  CF = 120MHz, cBW = 5.9kHz, nchans = 1024, tsamp = 165us
    /  dm = 15, snr = 10, period = 1s, width = 1%, tobs = 5min
    /
    /  SURVEY PARAMETERS
    /  Call1: loDM = 0;     hiDM = 21.12, dDM = 0.01, binsize = 1, dsubDM = 0.24, ndms = 2112, DMs/call = 24, ncalls = 88
    /  Call2: loDM = 21.12; hiDM = 36.96, dDM = 0.02, binsize = 2, dsubDM = 0.48, ndms = 792,  DMs/call = 24, ncalls = 33
    /  Call3: loDM = 36.96; hiDM = 65.56, dDM = 0.05, binsize = 4, dsubDM = 1.10, ndms = 572,  DMs/call = 22, ncalls = 22
    */

    // Hard code survey parameters, for now
    SURVEY *survey = (SURVEY *) malloc(sizeof(SURVEY));

    survey -> num_passes = 3;
    survey -> pass_parameters = (SUBBAND_PASSES *) malloc(3 * sizeof(SUBBAND_PASSES)) ;
    survey -> tdms = 2112 + 792 + 484;

    survey -> pass_parameters[0].lowdm      = 0;
    survey -> pass_parameters[0].highdm     = 21.12;
    survey -> pass_parameters[0].dmstep     = 0.01;
    survey -> pass_parameters[0].sub_dmstep = 0.24;
    survey -> pass_parameters[0].binsize    = 1;
    survey -> pass_parameters[0].ndms       = 2112;
    survey -> pass_parameters[0].calldms    = 24;
    survey -> pass_parameters[0].ncalls     = 88;

    survey -> pass_parameters[1].lowdm      = 21.12;
    survey -> pass_parameters[1].highdm     = 36.96;
    survey -> pass_parameters[1].dmstep     = 0.02;
    survey -> pass_parameters[1].sub_dmstep = 0.48;
    survey -> pass_parameters[1].binsize    = 2;
    survey -> pass_parameters[1].ndms       = 792;
    survey -> pass_parameters[1].calldms    = 24;
    survey -> pass_parameters[1].ncalls     = 33;

    survey -> pass_parameters[2].lowdm      = 36.96;
    survey -> pass_parameters[2].highdm     = 65.56;
    survey -> pass_parameters[2].dmstep     = 0.05;
    survey -> pass_parameters[2].sub_dmstep = 1.10;
    survey -> pass_parameters[2].binsize    = 4;
    survey -> pass_parameters[2].ndms       = 484;
    survey -> pass_parameters[2].calldms    = 22;
    survey -> pass_parameters[2].ncalls     = 22; 

    return survey;
}

