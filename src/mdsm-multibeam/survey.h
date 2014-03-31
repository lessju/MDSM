#ifndef SURVEY_H_
#define SURVEY_H_

#include "stdio.h"

#define NULLVALUE -999

typedef struct 
{
    // Beam parameters
    unsigned beam_id, gpu_id;
    float foff, fch1;
    
    // Dedispersion parameters
    float *dm_shifts;
    unsigned maxshift;

} BEAM;

typedef struct
{
    unsigned from, to;
} RANGE;

typedef struct {

    // Data parameters
    bool voltage;
    unsigned nbeams, npols, nchans, nsamp, nbits, nantennas;
    float    tsamp;

    unsigned last_nsamp; // For file mode only, proper handling of last data buffer

    // Beam parameters
    BEAM *beams;

    // Beamforming parameters
    bool apply_beamforming;

    // Brute Force parameters
    float    lowdm, dmstep;
    unsigned tdms;

    // Timing parameters
    double timestamp;
    double blockRate;

    // Input file for standalone mode
    FILE *fp;

    // Output parameters
    char      fileprefix[120], basedir[120];
    unsigned  secs_per_file;
    char      use_pc_time, single_file_mode;

    // Bandpass parameters
    float bandpass_mean, bandpass_std, bandpass_rmse;
    unsigned ncoeffs;

    // RFI parameters
    bool apply_rfi_clipper;
    float spectrum_thresh, channel_thresh;
    unsigned channel_block, num_masks;
    RANGE* channel_mask;

    // Detection parameters
    float detection_threshold;
    bool  apply_median_filter;
    bool  apply_detrending;

    // Clustering parameters
    bool apply_clustering, apply_classification;
    unsigned dbscan_min_points;
    float dbscan_time_range, dbscan_dm_range, dbscan_snr_range;
    float min_pulse_width;

    // Write parameters
    bool tbb_enabled;
    bool dump_to_disk;
    unsigned output_bits;
    unsigned output_compression;    

    // Number of GPUs which are used
    unsigned num_threads;
    unsigned *gpu_ids;
    unsigned num_gpus;

} SURVEY;

#endif
