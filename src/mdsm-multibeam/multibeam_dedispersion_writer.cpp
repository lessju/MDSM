// MDSM stuff
#include "multibeam_dedispersion_writer.h"
#include "cache_brute_force.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

#include "omp.h"

// Lookup table for log value used during the data encoding
float log_lookup_table[LOG_LOOKUP_LENGTH];

// Maximum value permissible (higher value are clipped)
float maximumValue = 0;

// Quantise power data
void quantise_power_data(FILE *fp, SURVEY *survey, float *data, unsigned nvalues)
{
    // Check if first time that a buffer is being dumped to disk
    if (maximumValue == 0)
    {
        // Calculate maximum value
        for(unsigned i = 0; i < nvalues; i++)
            maximumValue = (data[i] > maximumValue) ? data[i] : maximumValue;
        maximumValue *= 2; // Double the value to be able to cater for brighter input (might need improvement)

        // Build lookup table
        for(unsigned i = 0; i < LOG_LOOKUP_LENGTH; i++)
            log_lookup_table[i] = log10(1 + i * survey -> output_compression / (float) LOG_LOOKUP_LENGTH );
    }

    // Define some initial values for fast processing
    float Q = 1.0 / pow(2, survey -> output_bits);
    float inverse_Q = 1.0 / Q;
    float log_one_plus_mu = 1.0 / log10(1 + survey -> output_compression);
    float inverse_maxValue = 1.0 / maximumValue;
    unsigned bitrange = pow(2, survey -> output_bits);

    // Start encoding data
    // We want to interleave CPU processing and data buffering on disk
    // so we divide the input buffer into N parts (assume power of 2)
    unsigned char *encodedData = (unsigned char *) data;
    for(unsigned p = 0; p < ENCODING_WRITE_OVERLAP; p++)
    {
        for(unsigned i = 0; i < nvalues / ENCODING_WRITE_OVERLAP; i++)
        {
            unsigned index = nvalues / ENCODING_WRITE_OVERLAP * p + i;
            float datum = data[index] * inverse_maxValue * LOG_LOOKUP_LENGTH;
            datum = log_lookup_table[(int) datum] * log_one_plus_mu;
            encodedData[index] = (((int)(datum * inverse_Q) * Q + Q * 0.5) * bitrange);
        }  

        // Dump data to disk
        fwrite(encodedData, nvalues / ENCODING_WRITE_OVERLAP, sizeof(unsigned char), fp);
    }
}

// Quantise complex data
void quantise_complex_data(FILE *fp, SURVEY *survey, short *data, unsigned nvalues)
{
    // Check if first time that a buffer is being dumped to disk
    if (maximumValue == 0)
    {
        maximumValue = 32768;

       // Create the lookup table and calculate its values
        // We only need 32768 value to cover the entire range
        // for signed short values (the negative values are
        // just mirrored on the negative y-axis)
        for(unsigned i = 0; i < maximumValue; i++) 
            log_lookup_table[i] = log10(1 + i * survey -> output_compression / maximumValue);
    }

    // Define some initial values for fast processing
    unsigned bitrange = pow(2, survey -> output_bits);
    float Q = 1.0 / bitrange;
    float inverse_Q = 1.0 / Q;
    float log_one_plus_mu = 1.0 / log10(1 + survey -> output_compression);

    // Start encoding data
    // We want to interleave CPU processing and data buffering on disk
    // so we divide the input buffer into N parts (assume power of 2)
    unsigned char *encodedData = (unsigned char *) data;
    for(unsigned p = 0; p < ENCODING_WRITE_OVERLAP; p++)
    {
        for(unsigned i = 0; i < nvalues / ENCODING_WRITE_OVERLAP; i++)
        {
            unsigned index = nvalues / ENCODING_WRITE_OVERLAP * p + i;

            // Extract signs of real and imaginary parts
            char real_sign = (data[index*2]   < 0) ? -1 : 1;
            char imag_sign = (data[index*2+1] < 0) ? -1 : 1;

            // Encode the value using the log lookup table
            float real_datum = log_lookup_table[abs(data[index*2])]   * log_one_plus_mu;
            float imag_datum = log_lookup_table[abs(data[index*2+1])] * log_one_plus_mu;

            // Quantise values to 3 bits (+ sign bit)
            unsigned char real_quant = ((char) (real_datum * inverse_Q) * Q + Q * 0.5) * bitrange;
            unsigned char imag_quant = ((char) (imag_datum * inverse_Q) * Q + Q * 0.5) * bitrange;

            // Combine values and sign bits to for 8-bit representation:
            // [rsign rX rX rX isign iX iX iX]
            unsigned char value = (real_sign   & 0x80)        |
                                  ((real_quant & 0x07) << 4)  |
                                  ((imag_sign  & 0x80) >> 4)  |
                                  (imag_quant  & 0x07);
            encodedData[i] =  value;
        }

        // Dump data to disk
        fwrite(encodedData, nvalues / ENCODING_WRITE_OVERLAP, sizeof(unsigned char), fp);
    }   
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
        struct timeval tstart, end;
        long mtime, seconds, useconds;    
        gettimeofday(&tstart, NULL);

        // Dump to file, quantising data if required
        if (survey -> output_bits != 32)
            // Quantise data and dump to file
            quantise_power_data(fp, survey,  params -> writer_buffer, survey -> nbeams * survey -> nsamp * survey -> nchans);
        else
            // Dump directly to file
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
