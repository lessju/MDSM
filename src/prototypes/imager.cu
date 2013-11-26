#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include "cuComplex.h"
#include "time.h"
#include "cuda_runtime.h"
#include <iostream>

using namespace std;

struct test
    {
        float vu[2];
        float w;
        float visibility_x;
        float visibility_y;
        float empt[3];
    } ;


__global__ void test(struct test * vis, int vis_entries, int vis_per_block, cuComplex *convFunc, 
                     int support, int sampling, int wplanes, float wpow2increment, cuComplex *output, 
                     int dim_row, int dim_col, float inc_row, float inc_col)
{
    __shared__ struct test2
    {
        struct test testing;
    } tt;

    support = support / 2;

    // Let's first see which visibility data shall be taken care of
    __shared__ float pos_y,pos_x;
    __shared__ int loc_y, loc_x;
    __shared__ int off_y, off_x;
    __shared__ int wplane;

    float tmp;
    cuComplex add;
    
     int begin = (blockIdx.y*vis_per_block);
     int end = begin + vis_per_block - 5;
     if (begin>=vis_entries) return;  //This block is useless
     if (end>=vis_entries) end = vis_entries - 1;
     
    for (int vis_no = begin; vis_no <= end; vis_no++)
    {
        __syncthreads();
        
        if ((threadIdx.x == 0) && (threadIdx.y == 0))
        {
            tt= *(struct test2*) (vis + vis_no);

        // We first find in which wplane point is
        wplane = (int) round(sqrtf(fabs(wpow2increment * tt.testing.w))); //still need to understand the +1 abd offset.. I beleive it is 0
        
        if ( wplane > (wplanes -1) ) wplane = wplanes - 1;

           pos_y=(tt.testing.vu[0] / inc_row) + __int2float_rz(dim_row / 2);
           pos_x=(tt.testing.vu[1] / inc_col) + __int2float_rz(dim_col / 2);
           loc_y = __float2int_rz(pos_y); 
           loc_x = __float2int_rz(pos_x); 
           off_y = __float2int_rz(__int2float_rz(loc_y-pos_y)* __int2float_rz(sampling));
           off_x = __float2int_rz(__int2float_rz(loc_x-pos_x)* __int2float_rz(sampling));
        }

        __syncthreads();

        if (((loc_y-support)<0) || ((loc_y+support)>(dim_row-1)) || ((loc_x-support)<0) || ((loc_x+support)>(dim_col-1)))
            continue;  /// out of grid 
       
        //continue;
        for (int iy = (-support + threadIdx.y); //*17
                      iy <= support;
                      iy += blockDim.y)
        {   
//            int ix_orig=(-support+threadIdx.x);
//            int ix=ix_orig;
//            ix -= (loc_x + ix - threadIdx.x) % 64; //Important 
            
            for (int  ix = (-support + threadIdx.x); //*17
                           ix <= support;
                           ix += blockDim.x)
            {
//                if (ix < ix_orig) continue;
               int newloc_y=loc_y+iy;
               int newloc_x=loc_x+ix;
                
                //newloc[0]=loc[0]+iy;
                //newloc[1]=loc[1]+ix;
                //loc_y+=iy;
                //loc_x+=ix;
                
                int poss=(newloc_y)*dim_col+newloc_x;
                // poss=0;
                //int iloc_y=(sampling*(iy+support))+(sampling-1)+off[0];
               // int iloc_x=(sampling*(ix+support/)+(sampling-1)+off[1];
                 int iloc_y=(off_y+(support))*sampling+(iy+support);
                 int iloc_x=(off_x+(support))*sampling+(ix+support);
                
                //REVIEW above.... worthed to do an interesting note
                //if ((iloc_x<0)||(iloc_y<0)) continue;
                
                int convplaneside=(support*2+1)*sampling-1;
                //if ((iloc_x>convplaneside)||(iloc_y>convplaneside)) continue;
                cuComplex addvalue;
                int pss2=wplane*convplaneside*convplaneside+(iloc_y*convplaneside+iloc_x);
             //   if (threadIdx.x>100000) printf("%d",pss2);
                //pss2-=(pss2-threadIdx.x)%32; // no real big difference
                //cuComplex wt=*(convFunc+ pss2);   //To change for convFunc //To CHECK... Do we expect complex values
                cuComplex wt; 
                wt.x=1.0f;
                wt.y=1.0f;
                addvalue.x = tt.testing.visibility_x * wt.x-tt.testing.visibility_y* wt.y;
                addvalue.y = tt.testing.visibility_y * wt.x+tt.testing.visibility_x* wt.y;  //can be improved
                add.x+=addvalue.x;
                add.y+=addvalue.y;
                (output+poss)->x=addvalue.x;
                (output+poss)->y=addvalue.y;
////                output[threadIdx.y * blockDim.x + threadIdx.x] = addvalue;
//                tmp+=addvalue.x+addvalue.y;
                
                
//               atomicAdd(&((output+poss)->x),addvalue.x);
//               atomicAdd(&((output+poss)->y),addvalue.y);
//                
//                
            }
        }
               
        //atomicAdd(&(output)->x,tmp);
    }

//     __syncthreads();
        if(threadIdx.x == 0 && threadIdx.y == 0)
     output[blockIdx.y].x=tmp; 
//           atomicAdd(&((output)->x),tmp);
//            atomicAdd(&((output)->y),add.y);
     
}


int main(int argc, char *argv[])
{
    float timestamp;
    float  *data;
    cuComplex* grid;
    cudaEvent_t event_start,event_stop;
        // Initialise
        cudaSetDevice(0);
	cudaThreadSetCacheConfig(cudaFuncCachePreferShared);        
        // Allocate and generate buffers
    cudaMalloc((void **) &data, 240000 * 8 * sizeof(float));
    cudaMalloc((void **) &grid, 2016* 2016 * sizeof(cuComplex));

    cudaMemset(data, 0, 240000 * 8 * sizeof(float));

    cudaEventCreate(&event_start);
        cudaEventCreate(&event_stop);
        cout <<"Allocation ready"<<endl;
    cudaEventRecord(event_start, 0);
    dim3 threadsPerBlock;
    dim3 blocks;
    int b=1000; 
     threadsPerBlock.x=32;
     threadsPerBlock.y=32;
     threadsPerBlock.z=1;
     blocks.x=1;
     blocks.y=24000;
     blocks.z=1;
    
   test<<<blocks,threadsPerBlock>>>
             ((struct test*)data, 
             240000,
             240000 / blocks.y,
          //   b_x, b_y,
             NULL,
             529,
             4,
             128,
             1.2,
             grid, 
             2016,2016,
             6.56984f,6.56984f
             );
  
        cudaEventRecord(event_stop, 0);
        cout << "Now waiting"<<endl;
        cout.flush();
        cudaEventSynchronize(event_stop);
  
        cudaEventElapsedTime(&timestamp, event_start, event_stop);
        printf("Calculated in %f\n", timestamp);
}
