#include <cutil.h>
#include <cufft.h>
#include <sys/time.h>

// Transpose that effectively reorders execution of thread blocks along diagonals of the 
// matrix (also coalesced and has no bank conflicts)
// Here blockIdx.x is interpreted as the distance along a diagonal and blockIdx.y as 
// corresponding to different diagonals
// blockIdx_x and blockIdx_y expressions map the diagonal coordinates to the more commonly 
// used cartesian coordinates so that the only changes to the code from the coalesced version 
// are the calculation of the blockIdx_x and blockIdx_y and replacement of blockIdx.x and 
// bloclIdx.y with the subscripted versions in the remaining code

#define TILE_DIM    16
#define BLOCK_ROWS  16

__global__ void transposeDiagonal(float *idata, float *odata, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int blockIdx_x, blockIdx_y;

    // do diagonal reordering
    if (width == height) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    } else {
        int bid = blockIdx.x + gridDim.x * blockIdx.y;
        blockIdx_y = bid % gridDim.y;
        blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
    }    

    int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;  
    int index_in = xIndex + (yIndex)*width;

    xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex) * height;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
          tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];

    __syncthreads();

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
      odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
}

// ---------------------- Separate Polarisations Loop ------------------------------

// Separate X and Y polarisation into separate buffers (Blocksize == nsubs)
__global__ void seperateXYPolarisations(float *input, float *output, int nsamp, 
                                        int nchans)
{
    // Assign each thread block to one sample
    for(unsigned s = blockIdx.x; 
                 s < nsamp;
                 s += gridDim.x)  {
                 
        // Load X polarisation and save in output   
        output[s * nchans + threadIdx.x] = input[s * nchans * 2 + threadIdx.x];

        // Load Y polarisation and save in output   
        output[nsamp * nchans + s * nchans + threadIdx.x] = input[s * nchans * 2 + nchans + threadIdx.x];
    }
}

// Expand polarisations from 16-bit complex to 32-bit complex
__global__ void expandValues(float *input, float *output, int nvalues)
{
    // Assign each thread block to one sample
    for(int s = threadIdx.x + blockIdx.x * blockDim.x; 
            s < nvalues;
            s += gridDim.x * blockDim.x)  {
                 
        // Load polarisations and save in output
        int val = (int) input[s];
        output[s * 2]     = val && 65535;
        output[s * 2 + 1] = (val >> 16) && 65535;
    }
}

// ---------------------- Intensities Calculation Loop  ------------------------------
__global__ void calculate_intensities(cufftComplex *inbuff, float *outbuff, int nsamp, 
                                      int nsubs, int nchans, int npols)
{
    unsigned s, c, p;
    
    for(s = threadIdx.x + blockIdx.x * blockDim.x;
        s < nsamp;
        s += blockDim.x * gridDim.x)
    {
        // Loop over all channels
        for(c = 0; c < nsubs; c++) {
              
            float intensity = 0;
            cufftComplex tempval;
                
            // Loop over polarisations
            for(p = 0; p < npols; p++) {

                // Square real and imaginary parts
                tempval = inbuff[p * nsubs * nsamp + c * nsamp + s] ;
                intensity += tempval.x * tempval.x + tempval.y * tempval.y;
            }

            // Store in output buffer
            outbuff[(c * nchans + s % nchans) * (nsamp / nchans) + s / nchans ] = intensity;
        }
    }
}
// ------------------------------------------------------------------------------------

int nchans = 4, nsamp = 32768, nsubs = 512, npols = 2;
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
       else if (!strcmp(argv[i], "-nsubs"))
           nsubs = atoi(argv[++i]);
       else if (!strcmp(argv[i], "-npol"))
           npols = atoi(argv[++i]);
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
    unsigned i, j, k;

    // Initialise CUDA stuff
    cudaSetDevice(2);
    cudaEventCreate(&event_start); 
    cudaEventCreate(&event_stop); 
    float timestamp;
    
    cufftHandle plan;
    float *d_input, *input;
    float *output, *d_output;
    
    printf("nsamp: %d, nsubs: %d, nchans: %d, npols: %d\n", nsamp, nsubs, nchans, npols);

    // Initialise data 
    input  = (float *) malloc(nsubs * nsamp * npols * sizeof(float));
    output = (float *) malloc(nsubs * nsamp * npols * sizeof(float) * 2);

    // Enable for Intensities check
//    for(unsigned p = 0; p < npols; p++)
//        for(k = 0; k < nsubs; k++)
//            for (i = 0; i < nsamp / nchans; i++) 
//                for(j = 0; j < nchans; j++)
//                {
//                    input[p * nsubs * nsamp + k * nsamp + i * nchans + j].x = i;
//                    input[p * nsubs * nsamp + k * nsamp + i * nchans + j].y = i;
//                }

    // Enable for transpose check
    for (i = 0; i < nsamp; i++) 
        for(j = 0; j < npols; j++)
            for(k = 0; k < nsubs; k++)
                 input[i * nsubs * npols + j * nsubs + k] = (j + 1) * k;
    

    // Allocate and transfer data to GPU (nsamp * nchans * npols)
    cudaMalloc((void **) &d_input,  sizeof(float) * nsubs * nsamp * npols);
    cudaMalloc((void **) &d_output, sizeof(float) * nsubs * nsamp * npols * 2);
    
    cudaEventRecord(event_start, 0);
    cudaMemcpy(d_input, input, sizeof(float) * nsubs * nsamp * npols, cudaMemcpyHostToDevice);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Copied input to GPU in: %lfms\n", timestamp);
       
    // Separate X and Y intensities into output buffer
    cudaEventRecord(event_start, 0);
    seperateXYPolarisations<<<dim3(512, 1), nsubs >>>(d_input, d_output, nsamp, nsubs);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Separated X and Y pols in: %lfms\n", timestamp);
    
//    cudaMemcpy(input, d_output, sizeof(float) * nsubs * nsamp * npols, cudaMemcpyDeviceToHost);
//    for (i = 0; i < npols; i++) 
//        for (j = 0; j < nsamp; j++) 
//            for(k = 0; k < nsubs; k++)
//                    if (input[i * nsamp * nsubs + j * nsubs + k] != (i + 1) * k)
//                        printf("%d - %d - %d = %f : %d\n", i, j, k, input[i * nsamp * nsubs + j * nsubs + k], (i + 1) * k);
                        
    // Perform transpose for each polarisation
    cudaEventRecord(event_start, 0);
    transposeDiagonal<<<dim3(nsubs/TILE_DIM, nsamp/TILE_DIM), 
                        dim3(TILE_DIM,BLOCK_ROWS) >>>
                        (d_output, d_input, nsubs, nsamp);
    transposeDiagonal<<<dim3(nsubs/TILE_DIM, nsamp/TILE_DIM), 
                        dim3(TILE_DIM,BLOCK_ROWS) >>>
                        (d_output + nsamp * nsubs, d_input + nsamp * nsubs, nsubs, nsamp);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Performed transpose in: %lfms\n", timestamp);
    
//    cudaMemcpy(input, d_input, sizeof(float) * nsubs * nsamp * npols, cudaMemcpyDeviceToHost);
//    for (i = 0; i < npols; i++) 
//        for (j = 0; j < nsubs; j++) 
//            for(k = 0; k < nsamp; k++)
//                if (input[i * nsamp * nsubs + j * nsamp + k] != (i + 1) * j)
//                    printf("%d - %d - %d = %f : %d\n", i, j, k, input[i * nsamp * nsubs + j * nsamp + k], (i + 1) * j);

    cudaEventRecord(event_start, 0);
    expandValues<<<dim3(512, 1), 512 >>>(d_input, d_output, nsamp * nsubs * npols);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Expanded values in: %lfms\n", timestamp);
    
    // Create plan
    cufftPlan1d(&plan, nchans, CUFFT_C2C, nsubs * npols * (nsamp / nchans));

    // Execute FFT on GPU
    cudaEventRecord(event_start, 0);
    cufftExecC2C(plan, (cufftComplex *) d_output, (cufftComplex *) d_output, CUFFT_FORWARD);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Processed 1D FFT in: %lfms\n", timestamp);

    // Calculate intensity and perform transpose in memory
    cudaEventRecord(event_start, 0);
    calculate_intensities<<<dim3(nsamp / blocksize, 1), blocksize>>>
                          ((cufftComplex *) d_output, d_input, nsamp, nsubs, nchans, npols);
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&timestamp, event_start, event_stop);
    printf("Processed Intensities in: %lfms\n", timestamp);

   // Get result
//   cudaMemcpy(output, d_output, sizeof(float) * nsubs * nsamp, cudaMemcpyDeviceToHost);
//   for(i = 0; i < nchans * nsubs; i++)
//       for(k = 0; k < nsamp / nchans; k++)
//               if (output[i * nsamp / nchans + k] != 2 * npols * k*k) 
//                   printf("%d = %f\n", k, output[i * nsamp / nchans + k]);
//   printf("\n");
   
   // Clean up
   cufftDestroy(plan);
   cudaFree(d_input);
   cudaFree(d_output);
}
