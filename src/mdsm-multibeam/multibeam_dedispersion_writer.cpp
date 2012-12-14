// MDSM stuff
#include "multibeam_dedispersion_writer.h"

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/distributions/normal.hpp>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

#include "omp.h"

using boost::math::normal; // typedef provides default type is double.

// Quantise floating point data to 8 bits
void quantise_32to8_bits(FILE *fp, float *data, unsigned nsamp, unsigned nchans)
{
    // Initialise parameters
    normal s;
    float thresh = 6;
    float p = 2.0 / ( 1.0 * nsamp + 1.0), q = p;
    float c = 1 - 2 * (thresh * pdf(s, thresh) - (thresh * thresh - 1) * cdf(s, -thresh));

    // Set number of OpenMP threads
    int num_threads = 8;
	omp_set_num_threads(num_threads);

    // Loop over all frequences
    float temp1 = q / c;
    float temp2 = 1 - q;

    // Keep record of the mean and std of each channel to rescale output data
    float means[nchans], stddevs[nchans], mins[nchans];

    #pragma omp parallel for \
        shared(nchans, nsamp, data, thresh, temp1, temp2, num_threads, means)
    for(unsigned i = 0; i < nchans; i++)
    {
        // Initialise mean and std for current channel
        unsigned num = 128;
        float m = 0, s2 = 0, s = 0;
        for(unsigned k = 0; k < num; k++)
        {
            m += data[i * nsamp + k];
            s2 += data[i * nsamp + k] * data[i * nsamp + k];
        }

        m  /= num;
        s2 = s2 / num - m * m;
        s  = sqrt(s2);

        // Loop over all samples
        float min = 999999999;
        for(unsigned j = 0; j < nsamp; j++)
        {
            // New clipped value
            float val = (data[i * nsamp + j] - m) / s;
            val = (val > thresh) ? thresh : ((val < -thresh) ? -thresh : val);
            data[i * nsamp + j] = val;
            min = (min > val) ? val : min;

            // Update running mean and std;
            m  += p * s * val;
            s2 = temp2 * s2 + temp1 * s2 * val * val;
            s = sqrt(s2);
        }

        means[i]   = m;
        stddevs[i] = s;
        mins[i]    = min;
    }

    // Global minimum
    float min = 9999999;
    for(unsigned i = 0; i < nchans; i++)
        min = (mins[i] < min) ? mins[i] : min;
    
    // Invert sign of minimum
    min = -min;

    // Data values are now in the range [min < v < thresh] 
    unsigned char *quantised = (unsigned char *)  data;
    float factor = 255 / (thresh + min);
    for(unsigned i = 0; i < nchans * nsamp; i++)
        quantised[i] = (data[i] + min) * factor;

    fwrite(means, sizeof(float), nchans, fp);  // Write means
    fwrite(stddevs, sizeof(float), nchans, fp); // Write stddevs
    fwrite(quantised, nchans * nsamp, sizeof(unsigned char), fp); // Write thresholded, quantized data
}


// Write request buffer to file
void* write_to_disk(void* writer_params)
{
    WRITER_PARAMS* params = (WRITER_PARAMS *) writer_params;
    SURVEY *survey = params -> survey;
    time_t start = params -> start;
    FILE *fp = NULL;
    bool create_new_file = false;
    char *filename;

    printf("%d: Started writer thread\n", (int) (time(NULL) - start));

    // Processing loop
    while (true) 
    {
        // Poll mutex-locked variable until we have something to write to disk
        while(true)
        {
            pthread_mutex_lock(params -> writer_mutex);
            
            if (params -> data_available)
            {
                create_new_file = params -> create_new_file;
                filename = params -> filename;
                pthread_mutex_unlock(params -> writer_mutex);
                break;
            }
            pthread_mutex_unlock(params -> writer_mutex);
            sleep(0.2);
        } 

        // We have data to dump to disk... check if we need to create a new file
        if (create_new_file)
        {   
            // Close previous file, if any
            if (fp != NULL) fclose(fp);
                
            // Open new file
            if ((fp = fopen(filename, "wb")) == NULL)
            {
                fprintf(stderr, "Could not open file [%s] for writing\n", filename);
                continue;
            }
        }
        else if (fp == NULL)
        {
            fprintf(stderr, "File not specified, cannot dump data\n");
            continue;
        }

        // We have a valid file and data available. Write to disk
        // TODO: super-optimise
        struct timeval tstart, end;
        long mtime, seconds, useconds;    
        gettimeofday(&tstart, NULL);

        // Quantise data
        if (1)
            // Quantise data and dump to file
            quantise_32to8_bits(fp, params -> writer_buffer, survey -> nsamp, survey -> nchans);
        else
            fwrite(params -> writer_buffer, survey -> nbeams * survey -> nchans * survey -> nsamp, sizeof(float), fp);

        gettimeofday(&end, NULL);
        seconds  = end.tv_sec  - tstart.tv_sec;
        useconds = end.tv_usec - tstart.tv_usec;

        mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
        printf("%d: Dumped to file %s in %ld ms\n", (int) (time(NULL) - start), filename, mtime);

        // Done writing to file, reset parameters and wait for next buffer
        pthread_mutex_lock(params -> writer_mutex);
        params -> data_available = 0;
        pthread_mutex_unlock(params -> writer_mutex);

        // Stopping clause
        if (((WRITER_PARAMS *) writer_params) -> stop)
            break;
    }   

    printf("%d: Exited gracefully [writer]\n", (int) (time(NULL) - start));
    pthread_exit((void*) writer_params);
}
