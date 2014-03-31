// MDSM stuff
#include "multibeam_dedispersion_output.h"
#include "unistd.h"
#include "math.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// C++ stuff
#include <QFile>
#include <cstdlib>
#include <iostream>

// OpenMP
#include <omp.h>

// Clustering Class
#include "dbscan.h"

// ========================== FILE HERLPER ===================================
void create_files(FILE** fp, SURVEY* survey, double timestamp)
{
    // Single file mode 
    if (survey -> single_file_mode) 
        for (unsigned i = 0; i < survey -> nbeams; i++)
        {
            char beam_no[2];
            sprintf(beam_no, "%d", i);

            char pathName[256];
            strcpy(pathName, survey -> basedir);
            strcat(pathName, "/");
            strcat(pathName, survey -> fileprefix);
            if (survey -> nbeams > 1)
            {
                strcat(pathName, "_beam_");
                strcat(pathName, beam_no);
            }
            strcat(pathName, ".dat");
            
            if ((fp[i] = fopen(pathName, "w")) == NULL)
            {
                fprintf(stderr, "Invalid output file path: %s\n", pathName);
                exit(-1);
            }
        }
    else
        // Multiple-file mode
        for (unsigned i = 0; i < survey -> nbeams; i++)
            {
                char beam_no[2];
                sprintf(beam_no, "%d", i);

                char pathName[256];
                strcpy(pathName, survey -> basedir);
                strcat(pathName, "/");
                strcat(pathName, survey -> fileprefix);
                strcat(pathName, "_beam_");
                strcat(pathName, beam_no);
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

                if ((fp[i] = fopen(pathName, "w")) == NULL)
                {
                    fprintf(stderr, "Invalid output file path: %s\n", pathName);
                    exit(-1);
                }

                printf("Output file: %s\n", pathName);
            }
}

// Trigger TBB dump to disk when when a valid detection is encountered
void triggerTBB(SURVEY *survey, OUTPUT_PARAMS *params, float *input_buffer, double timestamp)
{
    // Keep looping until writer is available
    while (true)
    {
        pthread_mutex_unlock(params -> writer_mutex);
        if (!params -> writer_params -> data_available)
        {
            // Writer is idle, copy input data and start writing
            memcpy(params -> writer_buffer, input_buffer, survey -> nbeams * survey -> nsamp * survey -> nchans * sizeof(float));

            // Adjust file parameters
            params -> writer_params -> create_new_file = true;

            // Create file name
            char pathName[256];
            strcpy(pathName, survey -> basedir);
            strcat(pathName, "/");
            strcat(pathName, survey -> fileprefix);
            strcat(pathName, "_TBB_");

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
            sprintf(tempStr, "_%d", survey -> output_bits);
            strcat(pathName, tempStr);
            strcat(pathName, "bits.dat");

            // Set filename
            memcpy(&(params -> writer_params -> filename), &pathName, 256 * sizeof(char));

            // Notify data writer
            params -> writer_params -> data_available = 1;

            // Ready, unlock mutex and return
            pthread_mutex_unlock(params -> writer_mutex);

            break;
        }
    
        // Wait for a while
        usleep(10);
    }
}

// ========================== DETECTION FUNCTIONS ============================

// Process brute force dedispersion if that was chosen
void process_brute(FILE* output, float *buffer, SURVEY *survey, int read_nsamp, 
                   double timestamp, double blockRate, time_t start_time, int last_buffer)
{
    unsigned int j, k, l;
    
    // Calculate the mean and standard deviation if not provided externally
    double localmean = 0, localrms = 0, tmp_stddev = 0;
    if (!survey -> apply_detrending)
    {
        for(j = 0; j < read_nsamp * survey -> tdms; j++)
        {
            localmean  += buffer[j];
            tmp_stddev += pow(buffer[j], 2);
        }
        localmean /= read_nsamp * survey -> tdms;
        localrms = sqrt(tmp_stddev / (read_nsamp * survey -> tdms) - localmean * localmean);
    }
    else // Normalisation was performed on the GPU
        { localmean = 0; localrms = 1; }

    printf("%d: Mean: %f, Stddev: %f\n", (int) (time(NULL) - start_time), localmean, localrms);
        
    // Subtract DM mean from all samples and apply threshold
    for (k = 0; k < survey -> tdms; k++) 
    {
        int index = k * read_nsamp; 
        for(l = 0; l < read_nsamp; l++) 
        {
            // Detection threshold sigma filter
            float dm = survey -> lowdm + k * survey -> dmstep;
            if (buffer[index + l] - localmean >= localrms * survey -> detection_threshold) 
                  fprintf(output, "%lf, %f, %f\n", timestamp + l * blockRate, 
                                                   dm, (buffer[index + l]  - localmean) / localrms);
        }   
    }   

    fflush(output);
}

// Process brute force dedispersion if that was chosen
unsigned process_brute_clustering(FILE* output, float *buffer, SURVEY *survey, unsigned numClusters,
                                 int read_nsamp, double timestamp, double blockRate, time_t start_time,
                                 int last_buffer, int beam_num)
{
    unsigned int j, k, l;

    // Initialise Data Points vector
    vector<DataPoint> dataPoints;

    // Check whether this is the last buffer in file mode
    
    // Calculate the mean and standard deviation if not provided externally
    double localmean = 0, localrms = 0, tmp_stddev = 0;
    if (survey -> apply_detrending)
    {
        localmean = 0;
        localrms = 0;
    }
    else if(!last_buffer)
    {
        for(j = 0; j < read_nsamp * survey -> tdms; j++)
        {
            localmean  += buffer[j];
            tmp_stddev += pow(buffer[j], 2);
        }
        localmean /= read_nsamp * survey -> tdms;
        localrms = sqrt(tmp_stddev / (read_nsamp * survey -> tdms) - localmean * localmean);
    }   
    else
    {
        unsigned long num_samp = 0;
        for(j = 0; j < survey -> tdms; j++)
            for(k = 0; k < survey -> last_nsamp - (survey -> beams[beam_num]).dm_shifts[j]; k++)
            {
                localmean += buffer[j * read_nsamp + k];   
                tmp_stddev += pow(buffer[j * read_nsamp + k], 2);
                num_samp++;
            }

        localmean /= num_samp;
        localrms    = sqrt(tmp_stddev / num_samp - localmean * localmean);
    }

    printf("%d: Mean: %f, Stddev: %f\n", (int) (time(NULL) - start_time), localmean, localrms);
        
    // Subtract DM mean from all samples and apply threshold
    for (k = 0; k < survey -> tdms; k++) 
    {
        int index = k * read_nsamp; 
        for(l = 0; l < read_nsamp; l++) 
        {
            // Detection threshold sigma filter
            float dm = survey -> lowdm + k * survey -> dmstep;
            if (buffer[index + l] - localmean >= localrms * survey -> detection_threshold) 
            {   
                // Data point beyond defined thresholds, add to data points vector
                DataPoint point = {timestamp + l * blockRate, dm, (buffer[index + l]  - localmean) / localrms, 0, 0};
                dataPoints.push_back(point);
            }
        }   
    }

    // We have now created a list of data points which exceed the threshold
    // Initialise clustering with computed values
    DBScan clustering(survey, survey -> dbscan_time_range * survey -> blockRate, survey -> dbscan_dm_range, 
                              survey -> dbscan_snr_range, survey -> dbscan_min_points);

    // Cluster data points
    vector<Cluster*> clusters = clustering.performClustering(dataPoints);
    unsigned clustersFound = clusters.size();

    printf("%d. Found %ld data points with %d clusters\n", (int) (time(NULL) - start_time), dataPoints.size(), clustersFound);

    unsigned validClusters = 0;
    for(unsigned i = 0; i < clustersFound; i++)
        if (survey -> apply_classification)
        {
            // Classify current cluster
            float val = clusters[i] -> computeTransientProbability(survey -> lowdm, survey -> dmstep, survey -> tdms);

            // For now, just notify plotter that the cluster was classified as RFI
            if (val < 0.1)
            {
                vector<int> *indices = clusters[i] -> getIndices();
                for(j = 0; j < indices -> size(); j++)
                {
                    DataPoint dataPoint = dataPoints[(*indices)[j]];
                    fprintf(output, "%lf,%f,%f,%d,%d\n", dataPoint.time, dataPoint.dm, dataPoint.snr, 
                                                         dataPoint.cluster + numClusters, dataPoint.type);
                }
                validClusters++;
            }
            else
            {
                vector<int> *indices = clusters[i] -> getIndices();
                for(j = 0; j < indices -> size(); j++)
                {
                    DataPoint dataPoint = dataPoints[(*indices)[j]];
                    fprintf(output, "%lf,%f,%f,%d,%d\n", dataPoint.time, dataPoint.dm, dataPoint.snr, 
                                                         -2, dataPoint.type);
                }
                validClusters++;
            }
        }
        else  // Just dump clusters to disk
        {
            // Dump clusters
            vector<int> *indices = clusters[i] -> getIndices();
            for(j = 0; j < indices -> size(); j++)
            {
                DataPoint dataPoint = dataPoints[(*indices)[j]];
                fprintf(output, "%lf,%f,%f,%d,%d\n", dataPoint.time, dataPoint.dm, dataPoint.snr, 
                                                     dataPoint.cluster + numClusters, dataPoint.type);
            }
            validClusters++;
        }

    // Dump noise
    for(j = 0; j < dataPoints.size(); j++)
    {
        DataPoint dataPoint = dataPoints[j];
        if (dataPoint.cluster == -1)
        fprintf(output, "%lf,%f,%f,%d,%d\n", dataPoint.time, dataPoint.dm, dataPoint.snr, 
                                             dataPoint.cluster, dataPoint.type);
    }

    fflush(output);

    // Erase all cluster data
    clusters.clear();
    dataPoints.clear();

    return clustersFound;
}

// Process dedispersion output
// Separate files are created per beam
void* process_output(void* output_params)
{
    OUTPUT_PARAMS* params = (OUTPUT_PARAMS *) output_params;
    SURVEY *survey = params -> survey;
    unsigned i, iters = 0, loop_counter = 0, pnsamp = params -> survey -> nsamp;
    int ppnsamp = params -> survey-> nsamp;
    time_t start = params -> start, beg_read;
    double pptimestamp = 0, ptimestamp = 0;
    double ppblockRate = 0, pblockRate = 0;
    int ret, written_samples = 0;

    printf("%d: Started output thread\n", (int) (time(NULL) - start));

    // Set number of OpenMP threads
    omp_set_num_threads(params -> nthreads);

    // Create file structures and files
    FILE *fp[survey -> nbeams];

    // Single output file mode, create files
    if (survey -> single_file_mode) 
        create_files(fp, survey, pptimestamp);

    unsigned numClusters[params -> nthreads];
    memset(numClusters, 0, params -> nthreads * sizeof(unsigned));

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
            // Create new output files if required
            if (written_samples == 0 && !(survey -> single_file_mode)) 
                create_files(fp, survey, pptimestamp);

            // Get input buffer pointer for current dedispersed output if TBB is enabled
            float *input_buffer = (params -> input_buffer)[(loop_counter - 2) % MDSM_STAGES];

            beg_read = time(NULL);

            // Processed dedispersed time series (each beam processed by one OpenMP thread)
            #pragma omp parallel \
                shared (fp, params, survey, ppnsamp, pptimestamp, ppblockRate, start, numClusters)
            {
                unsigned threadId = omp_get_thread_num();

                // Check if we're processing the last buffer
                int lastBuffer = iters >= params -> iterations - 2;

                // Apply clustering if required            
                if (survey -> apply_clustering)
                {
                    // Threshold and cluster points
                    unsigned found = process_brute_clustering(fp[threadId], (params -> output_buffer)[threadId], survey, 
                                                              numClusters[threadId], ppnsamp, pptimestamp, ppblockRate, 
                                                              start, lastBuffer, threadId);
                    numClusters[threadId] += found;

                    // If the TBB is enabled and there are valid detection, dump the unprocessed data to disk
                    if (found > 0 && survey -> tbb_enabled)
                        triggerTBB(survey, params, input_buffer, pptimestamp);   
                }
                else
                    // Just cluster points and dump them to disk
                    process_brute(fp[threadId], (params -> output_buffer)[threadId], survey, ppnsamp, pptimestamp, 
                                                ppblockRate, start, lastBuffer);
            }
            
            written_samples += ppnsamp;
            printf("%d: Processed output %d [output]: %d\n", (int) (time(NULL) - start), loop_counter,
                   (int) (time(NULL) - beg_read));
            
            // If enough time samples were written to file close current open descriptors
            // and open new files in the next iteration
            if (!(survey -> single_file_mode) && written_samples * ppblockRate > survey -> secs_per_file) 
            { written_samples = 0; for (i = 0; i < survey -> nbeams; i++) fclose(fp[i]); }
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
