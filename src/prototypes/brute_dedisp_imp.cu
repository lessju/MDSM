#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "unistd.h"
#include "time.h"
#include "string.h"
#include <cutil_inline.h>

// Stores output value computed in inner loop for each thread
// __shared__ float shared[512];

// Stores temporary shift values
__constant__ float dm_shifts[4096];

// ---------------------- Optimised Dedispersion Loop  ------------------------------
__global__ void dedisperse_loop(float *outbuff, float *buff, int nsamp, int nchans, float tsamp,
                                float startdm, float dmstep, int maxshift)
{
    extern __shared__ float shared[];

    int c, s = threadIdx.x + blockIdx.x * blockDim.x;
    float shift_temp = (startdm + blockIdx.y * dmstep) / tsamp;
    
    for (s = threadIdx.x + blockIdx.x * blockDim.x; 
         s < nsamp; 
         s += blockDim.x * gridDim.x) {

        shared[threadIdx.x] = 0;
     
        for(c = 0; c < nchans; c++) {
            int shift = c * (nsamp + maxshift) + floor(dm_shifts[c] * shift_temp);
            shared[threadIdx.x] += buff[shift + s ];
        }

        outbuff[blockIdx.y * nsamp + s] = shared[threadIdx.x];
    }
}

// ----- Dodson's kernels  ----
__global__ void dispSearch_kernel(float* g_disp, float* g_data,
					float f0, float df, int fN,
					float dt, int tN,
					float dmin, float dmul)
{
	// get thread ids
	int i,j;
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	int id_max = gridDim.x*blockDim.x;
	// get dispersion measure index
	int d = blockIdx.y;
	// get corresponding dispersion measure and multiply by the constant
	float kdm = 4.15e15*dmin*powf(dmul,d);
	if (dmul<0) kdm = 4.15e15*(dmin-d*dmul);
	// get max frequency
	float fM = f0+df*(fN-1);
	// divide all lags between the threads
	for (i=id; i<tN; i=i+id_max)
	{
		// add up along the candidate dm
		float sum = 0.0;
		for(j=0;j<fN;j++) {
			// get physical frequency value
			float f = f0+df*j;
			// get physical time value
			float t = kdm*(1/(f*f)-1/(fM*fM));
			// get array time offset
			int tloc = i + floorf(t/dt);
			if ((0<=tloc)&&(tloc<tN))
			{
				// and add it up
				sum += g_data[j*tN+tloc];
			}
		}
		// write sum to output
		g_disp[d*tN+i] = sum;
	}

}

__global__ void dispSearch_block_kernel(float* g_disp, float* g_data, 
					int sS, float s_0,
					float step, int sN, 
	   				int tN, int dN, int d_idx)
{
  int j,N;float sum,i;

  int id = blockDim.x*blockIdx.x + threadIdx.x;

  sum=0;
   {
    for (i=0;i<sN;i++) 
      { N=floorf(s_0-step*i+0.5);
	N=(id+N);//%tN;
	//N+=(i+sS)*tN;
	if ((N<tN)&&(N>=0)) {
	  sum += g_data[(int) (N+(i+sS)*tN)];
	}
     } }

   g_disp[id+tN*(d_idx)]=floorf(s_0-step*sN+0.5);;//s_0+sN*step;

}

// -------------------------- Main Program -----------------------------------


float fch1 = 156, foff = -0.005859375, tsamp = 0.000165, dmstep = 0.02, startdm = 0;
int nchans = 1024, nsamp = 1024, tdms = 1024;
int gridsize = 128, blocksize = 128;

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
       else if (!strcmp(argv[i], "-gridsize"))
           gridsize = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-blocksize"))
           blocksize = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-tsamp"))
           blocksize = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-foff"))
           foff = -atof(argv[++i]);
       i++;
    }
}

// Fill buffer with data (blocking call)
void generate_data(float* buffer, int nsamp, int nchans)
{
    for(int i = 0; i < nsamp * nchans; i++)
        buffer[i] = 0.1;
}

// DM delay calculation
float dmdelay(float f1, float f2)
{
  return(4148.741601 * ((1.0 / f1 / f1) - (1.0 / f2 / f2)));
}

int main(int argc, char *argv[])
{
   float *input, *output, *d_input, *d_output;
   int maxshift, i, j;

   process_arguments(argc, argv);

    // Calculate temporary DM-shifts
    float *dmshifts = (float *) malloc(nchans * sizeof(float));
    for (unsigned i = 0; i < nchans; i++)
          dmshifts[i] = dmdelay(fch1 + (foff * i), fch1);

    // Calculate maxshift
    maxshift = ceil(dmshifts[nchans - 1] * (startdm + dmstep * tdms) / tsamp);

    // Allocate and initialise arrays
    input = (float *) malloc( (nsamp + maxshift) * nchans * sizeof(float));
    output = (float *) malloc( nsamp * tdms * sizeof(float));
    for(i = 0; i < nchans; i++)
        for(j = 0; j < nsamp + maxshift; j++) {
            input[i * (nsamp + maxshift) + j] = i;
         }

    // Initialise CUDA stuff
    cutilSafeCall( cudaSetDevice(1));
    cudaEvent_t event_start, event_stop;
    float timestamp, kernelTime;

    cudaEventCreate(&event_start); 
    cudaEventCreate(&event_stop);

   printf("nsamp: %d, nchans: %d, tsamp: %f, startdm: %f, dmstep: %f, tdms: %d, fch1: %f, foff: %f, maxshift: %d\n",
           nsamp, nchans, tsamp, startdm, dmstep, tdms, fch1, foff, maxshift);

    // Allocate CUDA memory and copy dmshifts
    cutilSafeCall( cudaMalloc((void **) &d_input, (nsamp + maxshift) * nchans * sizeof(float)));
    cutilSafeCall( cudaMalloc((void **) &d_output, nsamp * tdms * sizeof(float)));
    cutilSafeCall( cudaMemset(d_output, 0, nsamp * tdms * sizeof(float)));
    cutilSafeCall( cudaMemcpyToSymbol(dm_shifts, dmshifts, nchans * sizeof(int)) );

    time_t start = time(NULL);

    // Copy input to GPU
    cudaEventRecord(event_start, 0);
    cutilSafeCall( cudaMemcpy(d_input, input, (nsamp + maxshift) * nchans * sizeof(float), cudaMemcpyHostToDevice) );    
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied to GPU in: %lf\n", timestamp);

//    dim3 gridDim(nsamp / blocksize, tdms);  
//    cudaEventRecord(event_start, 0);
//    dedisperse_loop<<<gridDim, blocksize, 512>>>(d_output, d_input, nsamp, nchans, tsamp, startdm, dmstep, maxshift);
//    cudaEventRecord(event_stop, 0);
//    cudaEventSynchronize(event_stop);
//    cudaEventElapsedTime(&timestamp, event_start, event_stop);
//    printf("Processed in: %lf\n", timestamp);
//    kernelTime = timestamp;
//     printf("Performance: %lf Gflops\n", (nchans * tdms) * (nsamp * 1.0 / kernelTime / 1.0e6));

    cudaEventRecord(event_start, 0);
	dim3 block(128,1,1);
	dim3 grid(30,tdms,1);
    dispSearch_kernel<<<grid, block>>>(d_output, d_input, fch1, foff, nchans, tsamp, nsamp, startdm, dmstep); 
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Processed in: %lf\n", timestamp);
    kernelTime = timestamp;
    printf("Performance: %lf Gflops\n", (nchans * tdms) * ((nsamp - maxshift) * 1.0 / kernelTime / 1.0e6));

//// * @param g_disp Output DM-Lag space
//// * @param g_data Input data (assumed lag-contiguous, may need cornerTurn first)
//// * @param f0 frequency of the lowest channel
//// * @param df bandwidth per channel
//// * @param fN total number of channels
//// * @param dt time per sample
//// * @param tN total number of samples per channel
//// * @param dmin lowest dispersion measure
//// * @param dN number of dispersion measures (currently limited by grid dim)
//// * @param dmul multiplication factor for dispersion measures



    // Copy output from GPU
    cudaEventRecord(event_start, 0);
    cutilSafeCall( cudaMemcpy(output, d_output, nsamp * tdms * sizeof(float), cudaMemcpyDeviceToHost) );    
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied from GPU in: %lf\n", timestamp);

    int val = 0;
    for(i = 0; i < nchans; i++) val += i;

//    for(i = 0; i < tdms; i++)
//        for(j = 0; j < nsamp; j++)
//            if (output[i * nsamp + j] != val)
//                printf("Error: dm: %d nsamp: %d value:%f \n", i, j, output[i*nsamp+j]);

    printf("Total time: %d\n", (int) (time(NULL) - start));
}

