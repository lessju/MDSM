#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define NROWS       2048
#define NCOLS       1024         
#define NUM_THREADS 32

// Matrix addition kernel
void __global__ matrixAdd(float *A, float *B, float *C, int numRows, int numCols)
{
    unsigned int index;

    // Some checks
    if (blockIdx.y * blockDim.y + threadIdx.y >= numRows)
        return;

    if (blockIdx.x * blockDim.x + threadIdx.x >= numCols)
        return;

    index = (blockIdx.y * blockDim.y + threadIdx.y) * numCols +
             blockIdx.x * blockDim.x + threadIdx.x;

    C[index] = A[index] + B[index];
}

int main()
{
    unsigned i, j;

    // Check if matrix is large enough
    if (NROWS < NUM_THREADS || NCOLS < NUM_THREADS)
    {
        printf("%d or %d < %d\n", NROWS, NCOLS, NUM_THREADS);
        exit(0);
    }

    // Allocate memory on CPU
    float *A, *B, *C;
    A = (float *) malloc(NROWS * NCOLS * sizeof(float));
    B = (float *) malloc(NROWS * NCOLS * sizeof(float));
    C = (float *) malloc(NROWS * NCOLS * sizeof(float));

    // Allocate memory on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, NROWS * NCOLS * sizeof(float));
    cudaMalloc((void **) &d_B, NROWS * NCOLS * sizeof(float));
    cudaMalloc((void **) &d_C, NROWS * NCOLS * sizeof(float));

    // Initliase data
    for (i = 0; i < NROWS; i++)
        for (j = 0; j < NCOLS; j++)
            A[i * NCOLS + j] = B[i * NCOLS + j] = j;

    // Allocate CUDA event
	cudaEvent_t event_start, event_stop;
	float timestamp;
    cudaEventCreate(&event_start); 
	cudaEventCreate(&event_stop); 

    // Copy data to GPU
    cudaEventRecord(event_start, 0);   

    cudaMemcpy(d_A, A, NROWS * NCOLS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, NROWS * NCOLS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, NROWS * NCOLS * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Copied data to GPU in: %lf ms\n", timestamp);

    // Configure and launch kernel    
    cudaEventRecord(event_start, 0);  

    dim3 numThreads(NUM_THREADS, NUM_THREADS);
    dim3 blockSize(ceil(NCOLS / ((float) NUM_THREADS)), ceil(NROWS / ((float) NUM_THREADS))); 
    printf("Num threads: %d, blocks: (%d, %d)\n", NUM_THREADS, blockSize.x, blockSize.y);

    matrixAdd<<<blockSize, numThreads>>>(d_A, d_B, d_C, NROWS, NCOLS);

    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Performed matrix addition in: %lf ms\n", timestamp);

    // Wait for kernel
    cudaThreadSynchronize();

    // Get result
    cudaEventRecord(event_start, 0);  

    cudaMemcpy(C, d_C, NROWS * NCOLS * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&timestamp, event_start, event_stop);
	printf("Copied results from GPU in: %lf ms\n", timestamp);
    
    // Check result
    for (i = 0; i < NROWS; i++)
        for (j = 0; j < NCOLS; j++)
           if (C[i * NCOLS + j] != A[i * NCOLS + j] + B[i * NCOLS + j])
            {
                printf("ERROR: %d %d (%f != %f)\n", i, j, A[i * NCOLS + j] + B[i * NCOLS + j], C[i * NCOLS + j]);
//                exit(0);
            }

    printf("Success!\n");

}
