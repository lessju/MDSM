###############################################################################

CCC = g++

###############################################################################

CC = $(CCC) $(DFITS) $(DFFTW) -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64

NVCC 		:= nvcc
CUDA_PATH     	:= /usr/local/cuda
CUDA_SDK_PATH 	:= /usr/local/cudaSDK/C
PRESTO   := /home/lessju/Code/presto
INC		:= -I. -I${CUDA_SDK_PATH}/common/inc -I${CUDA_PATH}/include
LDFLAGS 	:= -L${CUDA_PATH}/lib -L${CUDA_PATH}/lib64 -L${CUDA_SDK_PATH}/lib \
	          -L${CUDA_SDK_PATH}/common/lib -L/lib \
        	  -lcuda -lcudart  -lm
	
############################ CUDA DEDISPERSE ###################################

dedispersion:
	$(NVCC) -o dedispersion dedispersion.cu $(LDFLAGS) $(INC) -L/lib -D DEDISP_DMS=$(DEDISP_DMS) -D DEDISP_THREADS=$(DEDISP_THREADS) --use_fast_math --gpu-architecture=sm_30 --ptxas-options=-v -O3 --use_fast_math -Xptxas -dlcm=ca
	rm -f *.cu.o *.linkinfo

################################################################################
