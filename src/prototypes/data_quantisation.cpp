#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/distributions/normal.hpp>

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "time.h"
#include "string.h"
#include "sys/time.h"
#include "file_handler.h"

#include "omp.h"

char *filename = "/data/Data/Medicina/B0329+54_Nov_06/catenated_file.dat";

int nchans = 512, nsamp = 65536;
float p, q, c, thresh = 6;
int num_threads = 8;

using boost::math::normal; // typedef provides default type is double.

void read_data(float *buffer, unsigned nsamp, unsigned nchans)
{
    // Read file
    float *tempBuff = (float *) malloc(nsamp * nchans * sizeof(float));
    FILE *fp = fopen(filename, "rb");
    
    // Read header
    read_header(fp);

    read_block(fp, 32, tempBuff, nchans * nsamp);
    fclose(fp);

    // Transpose data
    unsigned i, j;
    for(i = 0; i < nchans; i++)    
        for(j = 0; j < nsamp; j++)
            buffer[i * nsamp + j] = tempBuff[j * nchans + i];

    free(tempBuff);
}

// Process command-line parameters
void process_arguments(int argc, char *argv[])
{
    int i = 1;
    
    while((fopen(argv[i], "r")) != NULL)
        i++;

    while(i < argc) {
       if (!strcmp(argv[i], "-nchans"))
           nchans = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-nsamp"))
           nsamp = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-tresh"))
           nsamp = atof(argv[++i]);
       i++;
    }
}

// Fast square root
float inline fsqrt(float x) 
{
    float xhalf = 0.5f * x;  // To avoid division by 2

    // Cast float to int (bit-level casting)
    int i = *(int *) &x;          

    // With some black magic, this gives an approximation to the sqrt in integer mode
    // Hex value depends on the power (here -0.5, which needs to be )
    i = 0x5f3759df - (i >> 1);

    // Cast in to float (bit-level casting)
    x = *(float *) &i;

    // One iteration of Newton Raphson method
    return x * (1.5f - (xhalf * x * x));
}

// MAIN FUNCTION
int main(int argc, char* argv[])
{
    unsigned i, j, k;

    // Process arguments
    process_arguments(argc, argv);

    // Initialise parameters
    normal s;
    p = q = 2.0 / ( 1.0 * nsamp + 1.0);
    c = 1 - 2 * (thresh * pdf(s, thresh) - (thresh * thresh - 1) * cdf(s, -thresh));

    // Generate data
    float *data = (float *) malloc(nsamp * nchans * sizeof(float));

    // Read file
    read_data(data, nsamp, nchans);

    printf("Starting filter\n");
    struct timeval start, end;
    long mtime, seconds, useconds;    
    gettimeofday(&start, NULL);

	// Set number of OpenMP threads
	omp_set_num_threads(num_threads);

    // Loop over all frequences
    float temp1 = q / c;
    float temp2 = 1 - q;

    // Keep record of the mean and std of each channel to rescale output data
    float means[nchans], stddevs[nchans], mins[nchans];

    // Dump original data (for comparison)
    FILE *forig = fopen("huber_test_orig.dat", "wb");
    fwrite(data, sizeof(float), nsamp * nchans, forig);
    fclose(forig);
    

    #pragma omp parallel for \
        shared(nchans, nsamp, data, thresh, temp1, temp2, num_threads, means) \
        private(i, j)
    for(i = 0; i < nchans; i++)
    {
        unsigned j;

        // Initialise mean and std for current channel
        int num = 128;
        float m = 0, s2 = 0, s = 0;
        for(k = 0; k < num; k++)
        {
            m += data[i * nsamp + k];
            s2 += data[i * nsamp + k] * data[i * nsamp + k];
        }

        m  /= num;
        s2 = s2 / num - m * m;
        s  = sqrt(s2);

        // Loop over all samples
        float min = 999999999;
        for(j = 0; j < nsamp; j++)
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
    for(i = 0; i < nchans; i++)
        min = (mins[i] < min) ? mins[i] : min;
    
    // Invert sign of minimum
    min = -min;

    // Data values are now in the range [min < v < thresh] 
    unsigned char *quantised = (unsigned char *)  data;
    float factor = 255 / (thresh + min);
    for(i = 0; i < nchans * nsamp; i++)
        quantised[i] = (data[i] + min) * factor;

    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;

    mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
    printf("Filtering done. Time: %ld ms\n", mtime);

    // Perform the quantisation

    FILE *fp = fopen("huber_test.dat", "wb");
    fwrite(means, sizeof(float), nchans, fp);  // Write means
    fwrite(stddevs, sizeof(float), nchans, fp); // Write stddevs
    fwrite(quantised, sizeof(unsigned char), nsamp * nchans, fp); // Write thresholded, quantized data buffer
    fclose(fp);

}
