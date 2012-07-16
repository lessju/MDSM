#include <cutil.h>
#include <cufft.h>
#include <sys/time.h>

// ---------------------- Intensities Calculation Loop  ------------------------------

// One thread block per original subband (nsamp = nsamp * chansPerSubband)
__global__ __device__ void rfi_clipping(float *input, float *means, int nsamp, int nsubs)
{
    __shared__ float2 tempSums[1024];
    float mean, stddev;   // Store as registers to avoid bank conflicts in shared memory

    // Initial setup
    tempSums[threadIdx.x].x = 0;
    tempSums[threadIdx.x].y = 0;

    for(unsigned s =  threadIdx.x; 
                 s <  nsamp; 
                 s += blockDim.x)
    {
        // Calculate partial sums
        float intensity = input[blockIdx.y * nsamp + s];
        tempSums[threadIdx.x].x += intensity;
        tempSums[threadIdx.x].y += intensity * intensity;
    }

    // synchronise threads
    __syncthreads();

    // TODO: use reduction to optimise this part
    if (threadIdx.x == 0) {

        float val1, val2;
        for(unsigned i = 0; i < blockDim.x; i++) {
            val1 += tempSums[i].x;
            val2 += tempSums[i].y;
        }

        // Calculate mean and stddev
        mean   = val1 / nsamp;
        stddev = sqrtf((val2 - nsamp * mean * mean) / nsamp);
        means[blockIdx.y] = mean;

        // Store mean and stddev in tempSums
        tempSums[0].x = mean;
        tempSums[0].y = stddev;
    }

    // Synchronise threads
    __syncthreads();  
    mean = tempSums[0].x;
    stddev = tempSums[0].y;

    // Clip RFI within the subbands
    for(unsigned s =  threadIdx.x; 
                 s <  nsamp;    
                 s += blockDim.x)
    {
        float val = input[blockIdx.y * nsamp + s];
        float tempval = fabs(val - mean);
        if (tempval >= stddev * 4 || tempval <= stddev / 4)
            val = mean;

        __syncthreads();
        input[blockIdx.y * nsamp + s] = val;
    }
}

// ------------------------------------------------------------------------------------

int nsamp = 32, nsubs = 8;
int blocksize = 512;

// Process command-line parameters
void process_arguments(int argc, char *argv[])
{
    int i = 1;
    
    while((fopen(argv[i], "r")) != NULL)
        i++;

    while(i < argc) {
       if (!strcmp(argv[i], "-nsamp"))
           nsamp = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-nsubs"))
           nsubs = atoi(argv[++i]);
      else if (!strcmp(argv[i], "-blocksize"))
           blocksize = atoi(argv[++i]);
       i++;
    }
}

// -------------------------- Main Program -----------------------------------

int main(int argc, char *argv[]) 
{
    // Initialise stuff
    process_arguments(argc, argv);
    cudaEvent_t event_start, event_stop;

    // Initialise CUDA stuff
    cudaSetDevice(1);
    cudaEventCreate(&event_start); 
    cudaEventCreate(&event_stop); 
    float timestamp;
    
    float *input, *d_input, *means, *d_means;
    
    printf("nsamp: %d, nsubs: %d\n", nsamp, nsubs);

    // Initialise data 
    input  = (float *) malloc(nsubs * nsamp * sizeof(float));
    means  = (float *) malloc(nsubs * sizeof(float));

    for(unsigned i = 0; i < nsubs; i++)
        for(unsigned j = 0; j < nsamp; j++)
            input[i * nsamp + j] = j;   

    // Allocate and transfer data to GPU (nsamp)
    cudaMalloc((void **) &d_input,  sizeof(float) * nsubs * nsamp);
    cudaMalloc((void **) &d_means,  sizeof(float) * nsubs);

    cudaMemset(d_input, 0,  sizeof(float) * nsubs * nsamp);
    
    // Copy data to GPU
    cudaEventRecord(event_start, 0);
    cudaMemcpy(d_input, input, sizeof(float) * nsubs * nsamp, cudaMemcpyHostToDevice);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied input to GPU in: %lfms\n", timestamp);

    // Apply inter-subband clipping
    cudaEventRecord(event_start, 0);
    rfi_clipping<<<dim3(1, nsubs), blocksize, blocksize*4 >>>
                          (d_input, d_means, nsamp, nsubs);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Processed Intensities in: %lfms\n", timestamp);

    // Copy means to host memory
    cudaMemcpy(means, d_means, sizeof(float) * nsubs, cudaMemcpyDeviceToHost);
    for(unsigned i = 0; i < nsubs; i++) {
        if (means[i] != i)
            printf("Error: %d = %f\n", i, means[i]);
    }

    // Calculate mean of means
    double meanOfMeans;
    for(unsigned i = 0; i < nsubs; i++)
        meanOfMeans += means[i];
    meanOfMeans /= (nsubs * 1.0);
    printf("Mean of means: %lf\n", meanOfMeans);

    // Apply subband excision
    cudaEventRecord(event_start, 0);

    for(unsigned i = 0; i < nsubs; i++)
        if (means[i] > 2 * meanOfMeans)
            cudaMemset(d_input + i * nsamp, 0, nsamp * sizeof(float));

    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Subband RFI excision in: %lfms\n", timestamp);

    // Get result
    cudaMemcpy(input, d_input, sizeof(float) * nsubs * nsamp, cudaMemcpyDeviceToHost);
//    for(unsigned i = 0; i < nsubs; i++)
//        for(unsigned j=0; j < nsamp; j++)
//            printf("%d\t%d\t:  %f\n", i, j, input[i*nsamp+j]);
   
   // Clean up
   cudaFree(d_input);
}
