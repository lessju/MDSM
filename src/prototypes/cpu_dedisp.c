#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "unistd.h"
#include "time.h"
#include "string.h"
#include "sys/time.h"

float fch1 = 126, foff = -6, tsamp = 5e-6, dmstep = 0.065, startdm = 0;
int nchans = 128, nsamp = 1024, tdms = 128, threads = 1;

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
       else if (!strcmp(argv[i], "-dmstep"))
           dmstep = atof(argv[++i]);
       else if (!strcmp(argv[i], "-startdm"))
           startdm = atof(argv[++i]);
       else if (!strcmp(argv[i], "-tdms"))
           tdms = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-threads"))
           threads = atoi(argv[++i]);
       i++;
    }

    foff = foff / (float) nchans;
    tsamp = tsamp * nchans;
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

   printf("nsamp: %d, nchans: %d, tsamp: %f, startdm: %f, dmstep: %f, tdms: %d, fch1: %f, foff: %f\n",
           nsamp, nchans, tsamp, startdm, dmstep, tdms, fch1, foff);

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
   generate_data(input, nsamp + maxshift, nchans);

   struct timeval start, end;
   long mtime, seconds, useconds;    
   gettimeofday(&start, NULL);

   for(n = 0; n < tdms; n++) {
       for(s = 0; s < nsamp; s++) {
           value = 0;
           for(c = 0; c < nchans; c++)
               value += input[(s + dmshifts[n][c]) * nchans + c];
           output[s * tdms + n] = value;
       }
    }

    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;

    mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
    printf("Time: %ld\n", mtime);
}

