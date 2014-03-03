
NVCC 		:= nvcc
CUDA_PATH   := /usr/local/cuda-5.5
INC		    := -I. -I${CUDA_PATH}/include
LDFLAGS 	:= -L${CUDA_PATH}/lib -L${CUDA_PATH}/lib64 -L/lib -lcuda -lcudart  -lm

downfacor:
	$(NVCC) -o downsample downsample.cu $(LDFLAGS) $(INC) -L/lib --use_fast_math --gpu-architecture=sm_35 --ptxas-options=-v -O3 --use_fast_math -Xptxas -dlcm=ca -lineinfo
	rm -f *.cu.o *.linkinfo

rearrange:
	$(NVCC) -o rearrange rearrange.cu $(LDFLAGS) $(INC) -L/lib --use_fast_math --gpu-architecture=sm_35 --ptxas-options=-v -O3 --use_fast_math -Xptxas -dlcm=ca -lineinfo
	rm -f *.cu.o *.linkinfo
