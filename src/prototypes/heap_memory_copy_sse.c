#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "unistd.h"
#include "time.h"
#include "string.h"
#include "sys/time.h"

#define USE_SSE 1
#define PWR(X,Y) X*X + Y*Y

#if USE_SSE
    #include <xmmintrin.h>
#endif

int nbeams = 1, nchans = 1024, nsamp = 65536 * 4, hsamp = 128;

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
       else if (!strcmp(argv[i], "-nbeams"))
           nbeams = atoi(argv[++i]);
       i++;
    }
}

int main(int argc, char *argv[])
{
    float *buffer, *heap;

    process_arguments(argc, argv);

    printf("nbeams: %d, nsamp: %d, nchans: %d, hsamp: %d\n",
            nbeams, nsamp, nchans, hsamp);

    // Initialise input buffer
    #if USE_SSE
        if ((posix_memalign((void **) &buffer, 8, nsamp * nchans * nbeams * sizeof(float))) != 0)
            printf("Error allocating memory\n");
        if ((posix_memalign((void **) &heap, 8, hsamp * nchans * nbeams * sizeof(float))) != 0)
            printf("Error allocating memory\n");
    #else
        buffer = (float *) malloc( nsamp * nbeams * nchans * sizeof(float));
        heap = (float *) malloc( hsamp * nchans * nbeams * sizeof(float));

    #endif
    memset(heap, 1, hsamp * nchans * nbeams * sizeof(float));  

    struct timeval start, end;
    long mtime, seconds, useconds;    
    gettimeofday(&start, NULL);

    short *complexData = (short *) heap;
    for(unsigned b = 0; b < nbeams; b++)
        for(unsigned c = 0; c < nchans; c++)
            #if USE_SSE
            {
                __m128i *pHeap   = (__m128i *) (heap + b * nchans * nsamp + c * nsamp);
                __m128i *pBuffer = (__m128i *) (buffer + b * nchans * nsamp + c * nsamp);

                for(unsigned s = 0; s < hsamp / 4; s++)
                {
                    __asm__("movl %0, %%EAX\n\t"
                            "movl %1, %%EBX\n\t"
                            :  : "g"(pHeap), "g"(pBuffer) :);

/*                    __m128i heapCopy = *pHeap;*/
/*                    __m128i mask      = _mm_set_ps1(255);*/

/*/*                    PSRLW __m64 _mm_srli_pi16(__m64 m, int count)                    */
/*                    _mm_srl_epi32(heapCopy, 8);*/
/*                    _mm_and_ps(heapCopy, mask);*/

/*                    pHeap++;*/
/*                    pBuffer++;*/
                }
            }
            #else
                for(unsigned s = 0; s < hsamp; s++)
                {                
                    unsigned bufferIndex = b * nchans * nsamp + c * nsamp + s + times;
                    unsigned complexIndex = 2*(b * nchans * hsamp + c * hsamp + s);
                    buffer[bufferIndex] = PWR(complexData[complexIndex], complexData[complexIndex+1]);
                }
            #endif


    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;

    mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
    printf("Copied heap in: %ldms\n", mtime);
}

