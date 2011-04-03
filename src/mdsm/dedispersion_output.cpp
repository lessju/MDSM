// MDSM stuff
#include "dedispersion_output.h"
#include "unistd.h"
#include "math.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// C++ stuff
#include <cstdlib>
#include <iostream>
// Calculate the median of five numbers for a median filter
float medianOfFive(float n1, float n2, float n3, float n4, float n5){
  float *a = &n1, *b = &n2, *c = &n3, *d = &n4, *e = &n5;
  float *tmp;

  // makes a < b and b < d
  if(*b < *a){
    tmp = a; a = b; b = tmp;
  }

  if(*d < *c){
    tmp = c; c = d; d = tmp;
  }

  // eleminate the lowest
  if(*c < *a){
    tmp = b; b = d; d = tmp; 
    c = a;
  }

  // gets e in
  a = e;

  // makes a < b and b < d
  if(*b < *a){
    tmp = a; a = b; b = tmp;
  }

  // eliminate another lowest
  // remaing: a,b,d
  if(*a < *c){
    tmp = b; b = d; d = tmp; 
    a = c;
  }

  if(*d < *a)
    return *d;
  else
    return *a;

}


// Calaculate the mean and standard deviation for the data
// NOTE: Performed only on output of first thread
void mean_stddev(float *buffer, SURVEY *survey, int read_nsamp, time_t start_time)
{
    unsigned int i, j, iters, vals, mod_factor, shift = 0;
    double total, total2;
    float mean = 0, stddev = 0, mean2 = 0;

    for(i = 0; i < survey -> num_passes; i++) {

        // Calculate the total number of values
        vals = read_nsamp / survey -> pass_parameters[i].binsize 
               * (survey -> pass_parameters[i].ncalls / survey -> num_threads) 
               * survey -> pass_parameters[i].calldms;
        mod_factor = vals < 32 * 1024 ? vals : 32 * 1024;

        // Split value calculation in "kernels" to avoid overflows      
        // TODO: Join mean and stddev kernel in one loop  
        // Calculate the mean and stddev
        iters = 0;
        while(1) {
	  total  = 0;
	  total2  = 0;
	  for(j = 0; j < mod_factor; j++){
	    total += buffer[shift + iters * mod_factor + j];
	    total2 += pow(buffer[shift + iters * mod_factor + j], 2);
	  };
	  mean += (total / j);
	  mean2 += (total2 / j);
	  iters++;
	  if (iters * mod_factor + j >= vals) break;
        }
        mean /= iters;  // Mean for entire array
        stddev = sqrt(mean2/iters - pow(mean,2));

        // Store mean and stddev values in survey
        survey -> pass_parameters[i].mean = mean;
        survey -> pass_parameters[i].stddev = stddev;
        printf("%d: Mean: %f, Stddev: %f [pass %d]\n", (int) (time(NULL) - start_time), mean, stddev, i);
        shift += vals;
    }
}

// Apply tresholding using mean and stddev
void process_subband(float *buffer, FILE* output, SURVEY *survey, int read_nsamp, size_t size, double timestamp, double blockRate)
{
    unsigned int i = 0, thread, k, l, ndms, nsamp, shift = 0;
    float temp_val, startdm, dmstep, mean, stddev;

    for(thread = 0; thread < survey -> num_threads; thread++) {

        for(shift = 0, i = 0; i < survey -> num_passes; i++) {
	    int index_dm0 = size * thread + shift;
            // Calaculate parameters
            nsamp   = read_nsamp / survey -> pass_parameters[i].binsize;
            startdm = survey -> pass_parameters[i].lowdm + survey -> pass_parameters[i].sub_dmstep 
                      * (survey -> pass_parameters[i].ncalls / survey -> num_threads) * thread;
            dmstep  = survey -> pass_parameters[i].dmstep;
            ndms    = (survey -> pass_parameters[i].ncalls / survey -> num_threads) 
                      * survey -> pass_parameters[i].calldms;
	    //            mean    = survey -> pass_parameters[i].mean;
	    //            stddev  = survey -> pass_parameters[i].stddev;

            // Subtract dm mean from all samples and apply threshold
            for (k = 1; k < ndms; k++) {
	        int index_dm = index_dm0 + k * nsamp;
	        double total = 0.0, total2 = 0.0;
		for(l = 0; l < nsamp ; l++) {
		  buffer[index_dm + l] -= buffer[index_dm0 + l];
		  total += buffer[index_dm + l];
		  total2 += pow(buffer[index_dm + l], 2);
                }
		mean = total / nsamp;
		stddev = sqrt(total2/nsamp - pow(mean,2));
		if ( k == 100)
		  printf(" Mean:%f StdDev:%f ", mean,stddev);
                for(l = 0; l < nsamp - 5; l++) {
		  float themedian, a, b, c, d, e, thisdm;
		  //                    temp_val = buffer[index_dm + l] - mean;
                    a = buffer[index_dm + l + 0];
                    b = buffer[index_dm + l + 1];
                    c = buffer[index_dm + l + 2];
                    d = buffer[index_dm + l + 3];
                    e = buffer[index_dm + l + 4];
		    themedian = medianOfFive (a, b, c, d, e );
		    // std::cout << themedian << std::endl;
		    // 3.5 sigma filter
                    thisdm = startdm + k * dmstep;
                    if (themedian - mean >= stddev * 3.5 && thisdm > 1.0 )
                        fprintf(output, "%lf, %f, %f\n", 
                                timestamp + l * blockRate * survey -> pass_parameters[i].binsize,
                                thisdm, themedian); 
                }
	    }            
            shift += nsamp * ndms;
        }
        
    }
}

// Apply mean and stddev to apply thresholding
void process_brute(float *buffer, FILE* output, SURVEY *survey, int read_nsamp, size_t size, double timestamp, double blockRate, time_t start_time)
{
	unsigned int j, k, l, iters, vals, mod_factor;
	float mean = 0, stddev = 0, temp_val;
	double total;

	// Calculate the total number of values
	vals = read_nsamp * survey -> tdms;
	mod_factor = vals < 32 * 1024 ? vals : 32 * 1024;

	// Split value calculation in "kernels" to avoid overflows
	// TODO: Join mean and stddev kernel in one loop

	// Calculate the mean
	iters = 0;
	while(1) {
		total  = 0;
		for(j = 0; j < mod_factor; j++)
			total += buffer[iters * mod_factor + j];
		mean += (total / j);

		iters++;
		if (iters * mod_factor + j >= vals) break;
	}
	mean /= iters;  // Mean for entire array

	// Calculate standard deviation
	iters = 0;
	while(1) {

		total = 0;
		for(j = 0; j < mod_factor; j++)
			total += pow(buffer[iters * mod_factor + j] - mean, 2);
		 stddev += (total / j);

		 iters++;
		 if (iters * mod_factor + j >= vals) break;
	}
	stddev = sqrt(stddev / iters); // Stddev for entire array

    printf("%d: Mean: %f, Stddev: %f\n", (int) (time(NULL) - start_time), mean, stddev);

    // Subtract dm mean from all samples and apply threshold
	unsigned thread;
	int thread_shift = survey -> tdms * survey -> dmstep / survey -> num_threads;
	for(thread = 0; thread < survey -> num_threads; thread++) {
            for (k = 0; k < survey -> tdms / survey -> num_threads; k++) {
                for(l = 0; l < survey -> nsamp; l++) {
                    temp_val = buffer[size * thread + k * survey -> nsamp + l] - mean;
                    if (abs(temp_val) >= (stddev * 5) ){
                        fprintf(output, "%lf, %f, %f\n", timestamp + l * blockRate,
                                survey -> lowdm + (thread_shift * thread) + k * survey -> dmstep, temp_val + mean);
                    }   
                }
            }
        }
}

// Process dedispersion output
void* process_output(void* output_params)
{
    OUTPUT_PARAMS* params = (OUTPUT_PARAMS *) output_params;
    SURVEY *survey = params -> survey;
    int i, iters = 0, ret, loop_counter = 0, pnsamp = params -> survey -> nsamp;
    int ppnsamp = params -> survey-> nsamp;
    time_t start = params -> start, beg_read;
    double pptimestamp = 0, ptimestamp = 0;
    double ppblockRate = 0, pblockRate = 0;
    long written_samples = 0;
    FILE *fp = NULL;

    printf("%d: Started output thread\n", (int) (time(NULL) - start));

    // Single output file mode, create file
    if (survey -> single_file_mode) {
        char pathName[256];
        strcpy(pathName, survey -> basedir);
        strcat(pathName, "/");
        strcat(pathName, survey -> fileprefix);
        strcat(pathName, ".dat");
        fp = fopen(pathName, "w");
    }

    // Processing loop
    while (1) {

        // Wait input barrier
        ret = pthread_barrier_wait(params -> input_barrier);
        if (!(ret == 0 || ret == PTHREAD_BARRIER_SERIAL_THREAD)) 
            { fprintf(stderr, "Error during input barrier synchronisation [output]\n"); exit(0); }

        // Process output
        if (loop_counter >= params -> iterations) {

            // Create new output file if required
            if (written_samples == 0 && !(survey -> single_file_mode)) {
                char pathName[256];
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
                    time_t currTime = (time_t) pptimestamp;
                    tmp = localtime(&currTime);
                }       

                char tempStr[30];
                strftime(tempStr, sizeof(tempStr), "%F_%T", tmp);

                strcat(pathName, tempStr);
                strcat(pathName, "_");
                sprintf(tempStr, "%d", survey -> secs_per_file);
                strcat(pathName, tempStr);
                strcat(pathName, ".dat");

                fp = fopen(pathName, "w");
            }

            beg_read = time(NULL);
            if (params -> survey -> useBruteForce)
            	process_brute(params -> output_buffer, fp, params -> survey,  ppnsamp,
                              params -> dedispersed_size, pptimestamp, ppblockRate, start);
            else {
	      //				mean_stddev(params -> output_buffer, params -> survey, ppnsamp, start);
				printf("%d: Calculated mean and stddev %d [output]: %d\n", (int) (time(NULL) - start), loop_counter,
																		   (int) (time(NULL) - beg_read));
				process_subband(params -> output_buffer, fp, params -> survey,  ppnsamp,
						 	 	params -> dedispersed_size, pptimestamp, ppblockRate);
            }
            printf("%d: Processed output %d [output]: %d\n", (int) (time(NULL) - start), loop_counter,
            												 (int) (time(NULL) - beg_read));

            if (!(survey -> single_file_mode)) {
                written_samples += ppnsamp;
                if (written_samples * ppblockRate > survey -> secs_per_file) {
                    written_samples = 0;
                    fclose(fp);
                }
            }
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
        ppblockRate = pblockRate;
        pblockRate = params -> survey -> blockRate;    

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
