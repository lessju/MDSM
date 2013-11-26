
NVCC 		:= nvcc
CUDA_PATH   := /usr/local/cuda-5.5
INC		    := -I. -I${CUDA_PATH}/include
LDFLAGS 	:= -L${CUDA_PATH}/lib -L${CUDA_PATH}/lib64 -L/lib -lcuda -lcudart  -lm

dedispersion:
	$(NVCC) -o dedispersion dedispersion.cu $(LDFLAGS) $(INC) -L/lib --use_fast_math --gpu-architecture=sm_30 --ptxas-options=-v -O3 --use_fast_math -Xptxas -dlcm=ca -lineinfo
	rm -f *.cu.o *.linkinfo
