#include "dedispersion_output.h"
#include "survey.h"
#include "string.h"
#include "stdlib.h"
#include "stdio.h"
#include "unistd.h"
#include "math.h"
/*
void process_subbands(float **buffer, FILE* output, SURVEY *survey, int loop_counter)
{
    int tresh = 3600, i = 0, j, k, l, m, nsamp, ndms, ncalls, calldms, nsubs, tempval, shift = 0, binsize = survey -> pass_parameters[0].binsize; 
    float temp_val, mean, startdm, dmstep; 

    for(i = 0; i < survey -> num_passes; i++) {

        // Handle dedispersion maxshift 
        tempval = 3878;

        nsamp     = (survey -> nsamp + tempval) / binsize;
        startdm   = survey -> pass_parameters[i].lowdm;
        dmstep    = survey -> pass_parameters[i].sub_dmstep;
        ncalls    = survey -> pass_parameters[i].ncalls;
        calldms   = survey -> pass_parameters[i].calldms;
        nsubs     = survey -> nsubs;

        // Subtract dm mean from all samples and apply threshold        
        printf("nsamp: %d, shift: %d\n", nsamp, shift);
        for (k = 0; k < ncalls; k++) {  

            for(l = 0; l < nsamp; l++) {

                for (m = 0; m < nsubs; m++) {
                    temp_val = buffer[0][shift + k * nsubs * nsamp + l * nsubs + m];

                    if (temp_val >= tresh ) 
                       
                        fprintf(output, "%d, %.1f, %.1f\n", loop_counter * survey -> nsamp + l * binsize, startdm + k * dmstep, temp_val);
                }
            }
        }

        fflush(output);
        shift += nsamp * ncalls * nsubs;
        binsize *= 2;
    }
} */

void process(float **buffer, FILE* output, SURVEY *survey, int loop_counter, int read_nsamp)
{
    int tresh = 4000, i = 0, k, l, ndms, nsamp, shift = 0; 
    float temp_val, mean, startdm, dmstep; 

    for(i = 0; i < survey -> num_passes; i++) {

        nsamp   = read_nsamp / survey -> pass_parameters[i].binsize;
        startdm = survey -> pass_parameters[i].lowdm;
        dmstep  = survey -> pass_parameters[i].dmstep;
        ndms    = survey -> pass_parameters[i].ndms;

        // Subtract dm mean from all samples and apply threshold
        for (k = 1; k < ndms; k++) {
            mean = 0;

            for(l = 0; l < nsamp; l++)
                mean += buffer[0][shift + k * nsamp + l];

            mean /= (float) nsamp;
               
            for(l = 0; l < nsamp; l++) {
                temp_val = buffer[0][shift + k * nsamp + l] - mean;

                if (temp_val >= tresh )
                      fprintf(output, "%d, %f, %f\n", loop_counter * survey -> nsamp + l * survey -> pass_parameters[i].binsize, startdm + k * dmstep, temp_val);
            }
        }

        fflush(output);
        shift += nsamp * ndms;
    }
}


void* process_output(void* output_params)
{
    OUTPUT_PARAMS* params = (OUTPUT_PARAMS *) output_params;
    int i, iters = 0, ret, loop_counter = 0, pnsamp = params -> nsamp, ppnsamp = params -> nsamp;
    time_t start = params -> start, beg_read;

    // Allocate enough stack space
    printf("%d: Started output thread\n", (int) (time(NULL) - start));

    // Processing loop
    while (1) {

        // Wait input barrier
        ret = pthread_barrier_wait(params -> input_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD)) 
            { fprintf(stderr, "Error during input barrier synchronisation [output]\n"); exit(0); }

        // Process output
        if (loop_counter >= params -> iterations) {
            beg_read = time(NULL);
            process(params -> output_buffer, params -> output_file, params -> survey, loop_counter - params -> iterations, ppnsamp);
            printf("%d: Processed output %d [output]: %d\n", (int) (time(NULL) - start), loop_counter, 
                                                             (int) (time(NULL) - beg_read));
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
        pnsamp = params -> nsamp;         

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
