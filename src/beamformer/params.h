// Parameters modifiable by the user
#define BEAMS 32                 // Number of beams to generate
#define NTAPS 16                // Number of taps for PFB FIR
#define PLOT 0                  // Enable or disbale plotting


// Constant or Medicina-specific parameters
#define BEAMFORMER_THREADS 128
#define ANTS 32                 // Number of antennas
#define HEAP 128                // Specific to Medicina data output format

#if BEAMS < 16
   #define BEAMS_PER_TB BEAMS
#else
   #define BEAMS_PER_TB 16
#endif
