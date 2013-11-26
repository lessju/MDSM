#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/distributions/normal.hpp>

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "time.h"
#include "string.h"
#include "sys/time.h"

#include "omp.h"

int nchans = 128, nsamp = 1024;
float p, q, c, thresh = 8;

using boost::math::normal; // typedef provides default type is double.

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

    // Populate data with random data   
    boost::mt19937 rng;
    boost::normal_distribution<> nd(0.0, 1.0);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);

    for (i = 0; i < nchans; i++)
        for (j = 0; j < nsamp; j++)  
        {
            data[i * nsamp + j] = var_nor();
            if (j > 8192 && j < 8192 + 500)
                data[i * nsamp + j] += 1.5 * var_nor();
        }

    printf("Starting filter\n");
    struct timeval start, end;
    long mtime, seconds, useconds;    
    gettimeofday(&start, NULL);

    // Loop over all frequences
    float temp1 = q / c;
    float temp2 = 1 - q;

	// Set number of OpenMP threads
	omp_set_num_threads(8);

    #pragma omp parallel for \
        shared(nchans, nsamp, data, thresh, temp1, temp2) \
        private(i, j)
    for(i = 0; i < nchans; i++)
    {
        unsigned j;

        // Initialise mean and std for current channel
        float m = 0, s2 = 0, s = 0;
        for(k = 0; k < 64; k++)
        {
            m += data[i * nsamp + k];
            s2 += data[i * nsamp + k] * data[i * nsamp + k];
        }

        m  /= 64;
        s2 = s2 / 64 - m * m;
        s  = sqrt(s2);

        // Loop over all samples
        for(j = 0; j < nsamp; j++)
        {
            // New clipped value
            float val = (data[i * nsamp + j] - m) / s;
            val = (val > thresh) ? thresh : ((val < -thresh) ? -thresh : val);
            data[i * nsamp + j] = val;

            // Update running mean and std;
            m  += p * s * val;
            s2 = temp2 * s2 + temp1 * s2 * val * val;
            s = sqrt(s2);
//            printf("%f %f %f\n", m, s2, val);
        }
    }

    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;

    mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
    printf("Filtering done. Time: %ld ms\n", mtime);
    FILE *fp = fopen("huber_test.dat", "wb");
    fwrite(data, sizeof(float), nsamp * nchans, fp);
    fclose(fp);

}
