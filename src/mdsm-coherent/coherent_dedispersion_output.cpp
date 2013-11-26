// MDSM stuff
#include "coherent_dedispersion_output.h"
#include "unistd.h"
#include "math.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// C++ stuff
#include <cstdlib>
#include <iostream>

// pgplot
#include "cpgplot.h"
#define PLOT 0

// Process dedispersion output
void* process_output(void* output_params)
{
    OUTPUT_PARAMS* params = (OUTPUT_PARAMS *) output_params;
    OBSERVATION *obs = params -> obs;
    int i, iters = 0, ret, loop_counter = 0, pnsamp = params -> obs -> nsamp;
    int ppnsamp = params -> obs-> nsamp;
    time_t start = params -> start, beg_read;
    double pptimestamp = 0, ptimestamp = 0;
    double ppblockRate = 0, pblockRate = 0;

    // Initialise pg plotter
    #if PLOT
        if(cpgbeg(0, "/xwin", 1, 1) != 1)
            printf("Couldn't initialise PGPLOT\n");
        cpgask(false);
    #endif

    printf("%d: Started output thread\n", (int) (time(NULL) - start));

    FILE *fp = fopen("chanOutput.dat", "wb");

    // Processing loop
    while (1) {

        // Wait input barrier
        ret = pthread_barrier_wait(params -> input_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD)) 
            { fprintf(stderr, "Error during input barrier synchronisation [output]\n"); exit(0); }

        // Process output
        if (loop_counter >= params -> iterations) 
        {
            unsigned nsamp = ppnsamp, nchans = obs -> nchans;
            double timestamp = pptimestamp, blockRate = ppblockRate;
            beg_read = time(NULL);

            #if PLOT
            if (!obs -> folding)
            {
                // Process and plot output
                unsigned startChan = 0, endChan = startChan + 32, decFactor = 512;
//                unsigned startChan = 0, endChan = startChan + 32, decFactor = 4;
                float xr[nsamp / decFactor], yr[endChan - startChan][nsamp / decFactor];
                float ymin = 9e12, ymax=9e-12;

                for (unsigned c = 0; c < endChan - startChan; c++)
                {
                    // Decimate before plotting
                    for (unsigned i = 0; i < nsamp / decFactor; i++)
                    {        
                        unsigned index = (startChan + c) * nsamp + i * decFactor;
                        xr[i] = i;
                        yr[c][i] = 0;

                        for (unsigned j = 0; j < decFactor; j++)
                            yr[c][i] += params -> host_odata[index+j].x * 
                                        params -> host_odata[index+j].x + 
                                        params -> host_odata[index+j].y * 
                                        params -> host_odata[index+j].y;

//                            yr[c][i] = (yr[c][i] / decFactor) + c * 1e5;
                        yr[c][i] = (yr[c][i] / decFactor) + c * 4;
                        if (ymax < yr[c][i]) ymax = yr[c][i];
                        if (ymin > yr[c][i]) ymin = yr[c][i];
                    }
                }

                unsigned plotChan = 1;
                float *chan = (float *) malloc(nsamp * sizeof(float));
                for (unsigned i = 0; i < nsamp; i++)
                  chan[i] = params -> host_odata[plotChan * nsamp + i].x * 
                            params -> host_odata[plotChan * nsamp + i].x + 
                            params -> host_odata[plotChan * nsamp + i].y * 
                            params -> host_odata[plotChan * nsamp + i].y;

                fwrite(chan, sizeof(float), nsamp, fp);
                fflush(fp);
                free(chan);

                cpgenv(0.0, pnsamp / decFactor, ymin, ymax, 0, 1);
                cpgsci(7);
                for (unsigned i = 0; i < endChan - startChan; i++)
                    cpgline(nsamp / decFactor, xr, yr[i]);

                cpgmtxt("T", 2.0, 0.0, 0.0, "Dedispersed Channel Plot");
            }
            else
            {
                unsigned plotChannel = 15;
                unsigned decFactor = 256; 
                float *xr = (float *) malloc(obs -> profile_bins / decFactor * sizeof(float));
                float *yr = (float *) malloc(obs -> profile_bins / decFactor * sizeof(float));
                float ymin = 9e12, ymax = 9e-12;           
    
                unsigned fullProfiles = obs -> nsamp * (loop_counter - 1) / 
                                        obs -> profile_bins;
                unsigned leftover = (obs -> nsamp * (loop_counter - 1)) % obs -> profile_bins;

                if (fullProfiles > 0)
                {
                    // Decimate before plotting
                    for (unsigned i = 0; i < obs -> profile_bins / decFactor; i++)
                    {        
                        unsigned index = plotChannel * obs -> profile_bins + i * decFactor;
                        xr[i] = i;
                        yr[i] = 0;

                        for (unsigned j = 0; j < decFactor; j++)
                            yr[i] += params -> host_profile[index + j];

                        yr[i] = (yr[i] / decFactor);
                        yr[i] = (i < leftover / decFactor) 
                              ? yr[i] / (fullProfiles + 1) 
                              : yr[i] / fullProfiles;

                        if (ymax < yr[i]) ymax = yr[i];
                        if (ymin > yr[i]) ymin = yr[i];
                    }

                    cpgenv(0.0, obs -> profile_bins / decFactor, ymin, ymax, 0, 1);
                    cpgsci(7);
                    cpgline(obs -> profile_bins / decFactor, xr, yr);
                    cpgmtxt("T", 2.0, 0.0, 0.0, "Pulsar Profile");
                    
                    free(xr);
                    free(yr);
                }
            }
            #endif

            printf("%d: Processed output %d [output]: %d\n", (int) (time(NULL) - start), loop_counter,
                   (int) (time(NULL) - beg_read));
        }

    sleep(2);

        // Wait output barrier
        ret = pthread_barrier_wait(params -> output_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
            { fprintf(stderr, "Error during output barrier synchronisation [output]\n"); exit(0); }

        // Acquire rw lock
        if (pthread_rwlock_rdlock(params -> rw_lock))
            { fprintf(stderr, "Unable to acquire rw_lock [output]\n"); exit(0); } 

        // Update params
        ppnsamp = pnsamp;
        pnsamp = params -> obs -> nsamp;     
        pptimestamp = ptimestamp;
        ptimestamp = params -> obs -> timestamp;
        ppblockRate = pblockRate;
        pblockRate = params -> obs -> blockRate;    

        // Stopping clause
        if (((OUTPUT_PARAMS *) output_params) -> stop) {
            
            if (iters >= params -> iterations - 1) {
               
                // Release rw_lock
                if (pthread_rwlock_unlock(params -> rw_lock))
                    { fprintf(stderr, "Error releasing rw_lock [output]\n"); exit(0); }

                for(i = 0; i < params -> maxiters - params -> iterations; i++) {
                    pthread_barrier_wait(params -> input_barrier);
                    pthread_barrier_wait(params -> output_barrier);
                }
                break;
            }
            else
                iters++;
        }

        // Release rw_lock
        if (pthread_rwlock_unlock(params -> rw_lock))
            { fprintf(stderr, "Error releasing rw_lock [output]\n"); exit(0); }

        loop_counter++;
    }   

    printf("%d: Exited gracefully [output]\n", (int) (time(NULL) - start));
    pthread_exit((void*) output_params);
}
