// MDSM stuff
#include "beamforming_output.h"
#include "unistd.h"
#include "math.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

// C++ stuff
#include <QFile>
#include <cstdlib>
#include <iostream>


// ========================== FILE HERLPER ===================================
FILE* create_files(FILE* fp, SURVEY* survey, double timestamp)
{
    // We want to create one file with all the data contained within it
    // Data order will be time/channel/beam
    // We need a header so as to ease post-processing 

    // Generate filename
    char pathName[256];

    if (survey -> test)
    {
        strcpy(pathName, survey -> basedir);
        strcat(pathName, "/");
        strcat(pathName, survey -> fileprefix);
        strcat(pathName, "_test.dat");

    }
    else
    {
        strcpy(pathName, survey -> basedir);
        strcat(pathName, "/");
        strcat(pathName, survey -> fileprefix);
        strcat(pathName, "_");

        // Format timestamp 
        struct tm *tmp;
        if (survey -> use_pc_time) {
            time_t currTime = time(NULL);
            tmp = localtime(&currTime);                    
        }
        else {
            time_t currTime = (time_t) timestamp;
            tmp = localtime(&currTime);
        }       

        char tempStr[30];
        strftime(tempStr, sizeof(tempStr), "%F_%T", tmp);
        strcat(pathName, tempStr);
        strcat(pathName, ".dat");
    }

    // Create file
    if ((fp = fopen(pathName, "wb")) == NULL)
    {
        fprintf(stderr, "Invalid output file path: %s\n", pathName);
        exit(-1);
    }

    // Write file header
//    fprintf(fp, "nchans=%d,nbeams=%d,tsamp=%.6f\n", survey -> nchans, survey -> nbeams, survey -> tsamp * survey -> downsample);

    return fp;

}

// Process dedispersion output
// Separate files are created per beam
void* process_output(void* output_params)
{
    OUTPUT_PARAMS* params = (OUTPUT_PARAMS *) output_params;
    SURVEY *survey = params -> survey;
    unsigned i, iters = 0, loop_counter = 0, pnsamp = params -> survey -> nsamp;
    int ppnsamp = params -> survey -> nsamp;
    time_t start = params -> start;
    double pptimestamp = 0, ptimestamp = 0;

    int ret;
    FILE *fp = NULL;

    printf("%d: Started output thread\n", (int) (time(NULL) - start));

    // Processing loop
    while (1) 
    {
        // Wait input barrier
        ret = pthread_barrier_wait(params -> input_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD)) 
            { fprintf(stderr, "Error during input barrier synchronisation [output]\n"); exit(0); }

        // Process output
        if (loop_counter >= params -> iterations ) 
        {
            // Start timing
            struct timeval start_t, end_t;
            long mtime, seconds, useconds;    
            gettimeofday(&start_t, NULL);

            // First iteration, create file
            if (loop_counter == params -> iterations)
                fp = create_files(fp, survey, pptimestamp);
        
            if (survey -> perform_channelisation)
            {
                unsigned curr_nchans = survey -> subchannels * (survey -> stop_channel - survey -> start_channel);
                unsigned curr_nsamp = ppnsamp / survey -> subchannels;
                
                for(unsigned i = 0; i < curr_nsamp; i++)
                    fwrite((params -> output_buffer)[0] + i * curr_nchans * survey -> nbeams, 
                             sizeof(float), curr_nchans * survey -> nbeams, fp);
            }
            else
            {
                // Calculate sampling time, nsamp
                unsigned curr_nsamp = ppnsamp / survey -> downsample;

                // Write beam output to file in sample/channel/beam order (output buffer should be in this format)
                // Output is split into nthreads buffers, with beams split quasi-evenly among them
                // NOTE: we assume one GPU for now
                for(unsigned i = 0; i < curr_nsamp; i++)
                    fwrite((params -> output_buffer)[0] + i * survey -> nchans * survey -> nbeams, 
                             sizeof(float), survey -> nchans * survey -> nbeams, fp);
            }

            fflush(fp);
            fsync(fileno(fp));

            gettimeofday(&end_t, NULL);
            seconds  = end_t.tv_sec  - start_t.tv_sec;
            useconds = end_t.tv_usec - start_t.tv_usec;
            mtime = ((seconds) * 1000 + useconds / 1000.0) + 0.5;

            printf("%d: Processed Output %d: %ld ms\n", (int) (time(NULL) - start), loop_counter, mtime);
//            exit(0);
        }

        // Wait output barrier
        ret = pthread_barrier_wait(params -> output_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD))
            { fprintf(stderr, "Error during output barrier synchronisation [output]\n"); exit(0); }

        // Acquire rw lock
        if (pthread_rwlock_rdlock(params -> rw_lock))
            { fprintf(stderr, "Unable to acquire rw_lock [output]\n"); exit(0); } 

        // Update params
        ppnsamp = pnsamp;
        pnsamp = params -> survey -> nsamp;     
        pptimestamp = ptimestamp;
        ptimestamp = params -> survey -> timestamp;
//        ppblockRate = pblockRate;
//        pblockRate = params -> survey -> blockRate;  

        // Stopping clause
        if (((OUTPUT_PARAMS *) output_params) -> stop) {
            if (iters >= params -> iterations - 2) {
                // Release rw_lock
                if (pthread_rwlock_unlock(params -> rw_lock))
                    { fprintf(stderr, "Error releasing rw_lock [output]\n"); exit(0); }

                for(i = 0; i < params -> maxiters - params -> iterations + 1; i++) {            
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
