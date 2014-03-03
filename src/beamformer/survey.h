#ifndef SURVEY_H_
#define SURVEY_H_

#include "/usr/local/cuda-5.5/include/vector_types.h"
#include "stdio.h"

typedef struct 
{
    // Beam parameters
    unsigned beam_id, gpu_id;
    float foff, fch1, ra, dec, ha;
} BEAM;

typedef struct {

    // Data parameters
    unsigned int nsamp, nchans, nbits, nbeams, nantennas;
    float tsamp, foff, fch1;

    // Beam parameters
    float2 *beam_shifts;
    BEAM *beams;

    // Output parameters
    char      fileprefix[80], basedir[120];
    unsigned  secs_per_file;
    char      use_pc_time, single_file_mode;
    bool      test;

    // Channelisation parameters
    char fir_path[120];
    bool perform_channelisation, apply_pfb;
    unsigned int subchannels, ntaps;
    unsigned start_channel, stop_channel;
      
    // Timing parameters
    double timestamp;
    double blockRate;

    // Downsampling parameters
    unsigned downsample;

    // Number of GPUs which are used
    unsigned num_threads;
    unsigned *gpu_ids;
    unsigned num_gpus;

} SURVEY;

#endif
