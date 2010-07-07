#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "unistd.h"

float fch1 = 120, foff = -0.195, tsamp = 0.000661, dmstep = 0.065, startdm = 0;
int nchans = 512, nsamp = 32 * 1024, tdms = 1200;

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
       i++;
    }
}

// Fill buffer with data (blocking call)
int generate_data(float* buffer, int nsamp, int nchans)
{
    int i = 0;

    for(i = 0; i < nsamp * nchans; i++)
        buffer[i] = 0.1;

    return nsamp;
}

// DM delay calculation
float dmdelay(float f1, float f2)
{
  return(4148.741601 * ((1.0 / f1 / f1) - (1.0 / f2 / f2)));
}

int main(int argc, char *argv[])
{
   float *input, *output, value;
   int counter, s, c, n, maxshift;

   process_arguments(argc, argv);

    // Calculate DM-shifts
    int **dmshifts = (int **) malloc(tdms * sizeof(int *));
    for (n = 0; n < tdms; n++) {
        dmshifts[n] = (int *) malloc(nchans * sizeof(int));
        for (c = 0; c < nchans; c++)
            dmshifts[n][c] = (int) (dmdelay(fch1 + (foff * c), fch1) * (startdm + n * dmstep) / tsamp);
    }

    // Calculate maxshift
    maxshift = dmshifts[tdms - 1][nchans - 1];

    // Initialise input buffer
    input = (float *) malloc( (nsamp + maxshift) * nchans * sizeof(float));
    output = (float *) malloc( nsamp * tdms * sizeof(float));

   // Dedisperse
   time_t start = time(NULL);
   for(counter = 0; counter < 2076923; counter+= nsamp) {

       generate_data(input, nsamp + maxshift, nchans);
       printf("%d: %f done\n", (int) (time(NULL) - start), 100 * nsamp * counter / (double) (2076923 ));
       for(n = 0; n < tdms; n++)
           for(s = 0; s < nsamp; s++) {

               value = 0;
               for(c = 0; c < nchans; c++)
                   value += input[(s + dmshifts[n][c]) * nchans + c];
               output[s * tdms + n] = value;
           }
    }
    printf("Time: %d\n", (int) (time(NULL) - start));
}

