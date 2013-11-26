#include <cutil_inline.h>

// Stores output value computed in inner loop for each thread
#define FERMI
#ifdef FERMI
    __shared__ float localvalue[5000];
#else
    __shared__ float localvalue[4002];
#endif

// Stores temporary shift values and the list of dm values
__constant__ float dm_shifts[4096];

// The dedispersion loop
__global__ void dedisperse_subband(float *outbuff, float *buff, int nsamp, int nchans, int nsubs, 
                                   float startdm, float dmstep, int ndms, float tsamp)
{
    int samp, s, c, indx, soffset, sband, tempval, chans_per_sub = nchans / nsubs;
    float shift_temp;

    /* dedispersing loop over all samples in this buffer */
    s = threadIdx.x + blockIdx.x * blockDim.x;

    shift_temp = (startdm + blockIdx.y * dmstep) / tsamp;
    for (samp = 0; s + samp < nsamp; samp += blockDim.x * gridDim.x) {
        soffset = (s + samp);       

//        /* clear array elements for storing dedispersed subband */
        for (sband = 0; sband < nsubs; sband++)
            localvalue[threadIdx.x * nsubs + sband] = 0.0;

        /* loop over the subbands */
        for (sband = 0; sband < nsubs; sband++) {  
            tempval = (int) (dm_shifts[sband * chans_per_sub] * shift_temp); 
            for (c = (sband * chans_per_sub); c < (sband + 1) * chans_per_sub; c++) {
                indx = (soffset + (int) (dm_shifts[c] * shift_temp - tempval)) * nchans + c;
                localvalue[threadIdx.x * nsubs + sband] += buff[indx];
            }
        }

//        // Sync threads and store values in global memory
        for (sband = 0; sband < nsubs; sband++)
            outbuff[blockIdx.y * nsamp * nsubs + soffset * nsubs + sband] = localvalue[threadIdx.x * nsubs + sband];
    }
}

int nsamp = 8 * 1024, nchans = 4096, sdm = 5, nsubs = 32;
float tsamp = 0.005, fch1 = 120, foff = -0.0059;

// DM delay calculation
float dmdelay(float f1, float f2)
{
  return(4148.741601 * ((1.0 / f1 / f1) - (1.0 / f2 / f2)));
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
       else if (!strcmp(argv[i], "-sdm"))
           sdm = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-nsubs"))
           nsubs = atoi(argv[++i]);
       i++;
    }
}

int main(int argc, char *argv[])
{
    float *input, *output, *d_input, *d_output;
    int i, j, startdm = 0, dmstep = 1, ndms = 64;

    printf("hagu: %d\n", nsubs * 128);

    process_arguments(argc, argv);
    printf("nsamp: %d, nchans: %d, nsubs: %d, fch1: %f, foff: %f, tsamp: %f\n", nsamp, nchans, nsubs, fch1, foff, tsamp);

    // Calculate Subband DM-shifts
    float *dmshifts = (float *) malloc(nchans * sizeof(float));

    for (i = 0; i < nchans; i++)
          dmshifts[i] = dmdelay(fch1 + (foff * i), fch1);

    // Calculate maxshift (maximum for all threads)
    int maxshift = (int) (dmshifts[nchans - 1] * (startdm + ndms * dmstep) / tsamp);  
    printf("Maxshift: %d, %f\n", maxshift, dmshifts[nchans - 1]);

    // Allocate arrays
    output = (float *) malloc( nsamp * nsubs * ndms * sizeof(float));
    input  = (float *) malloc( (nsamp + maxshift) * nchans * sizeof(float));
    for (i = 0; i < nsamp + maxshift; i++)  
        for (j = 0; j < nchans; j++)
            input[i * nchans + j] = i;

    cutilSafeCall( cudaSetDevice(0));
    cudaEvent_t event_start, event_stop;
    float timestamp;
    int gridsize = 64;
    dim3 gridDim(gridsize, ndms);  

    cudaEventCreate(&event_start); 
    cudaEventCreate(&event_stop); 

    cutilSafeCall( cudaMalloc((void **) &d_input, (nsamp + maxshift) * nchans * sizeof(float)));
    cutilSafeCall( cudaMalloc((void **) &d_output, nsamp * nsubs * ndms * sizeof(float)));
    cutilSafeCall( cudaMemset(d_output, 0, nsamp * nsubs * sizeof(float)));
    cutilSafeCall( cudaMemcpy(d_input, input, (nsamp + maxshift) * nchans * sizeof(float), cudaMemcpyHostToDevice) );
    cutilSafeCall( cudaMemcpyToSymbol(dm_shifts, dmshifts, nchans * sizeof(int)) );

    cudaEventRecord(event_start, 0);
    dedisperse_subband<<< gridDim, 128 >>>(d_output, d_input, nsamp, nchans, nsubs, startdm, dmstep, ndms, tsamp);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Subbands processed in: %lf\n", i, timestamp);

    cutilSafeCall(cudaMemcpy(output, d_output, nsamp * nsubs * sizeof(float), cudaMemcpyDeviceToHost) );
//    for(i = 0; i < nsamp; i++ )
//        for (j = 0; j < nsubs; j++)
//            if ( output[i * nsubs + j] != nchans/nsubs);
//                printf("Error: samp: %d, chan: %d, value: %d\n", i, j, (int) output[i * nsubs + j]);

//    for(i = 0; i < nsamp; i++ ) {
//        for (j = 0; j < nsubs; j++)
//            printf("%d ", (int) output[i * nsubs + j]);
//        printf("\n");
//    }
}
