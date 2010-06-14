###############################################################################

BIN = /home/lessju/sigproc/cuda_dedispersion
CCC = gcc -O2

###############################################################################

CC = $(CCC) $(DFITS) $(DFFTW) -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64

NVCC 		:= nvcc
CUDA_PATH     	:= /usr/local/cuda
CUDA_SDK_PATH 	:= /home/lessju/NVIDIA_GPU_Computing_SDK/C
INC		:= -I. -I${CUDA_SDK_PATH}/common/inc -I${CUDA_PATH}/include
LDFLAGS 	:= -L${CUDA_PATH}/lib -L${CUDA_PATH}/lib64 -L${CUDA_SDK_PATH}/lib \
	          -L${CUDA_SDK_PATH}/common/lib \
        	  -lcuda -lcudart -lcutil
	
############################ CUDA DEDISPERSE ###################################

cudadedisperse :
	$(NVCC) -o main dedispersion_manager.cu --gpu-architecture sm_13 $(LDFLAGS) $(INC) -L/lib -lpthread -lm 
	rm -f *.cu.o *.linkinfo

binner:
	$(NVCC) -o binner binner.cu $(LDFLAGS) $(INC) -L/lib
	rm -f *.cu.o *.linkinfo


fft:
	$(NVCC) -o fft cuda_fft.cu $(LDFLAGS) $(INC) -L/lib -lcufft
	rm -f *.cu.o *.linkinfo

cpu_dedisp:
	$(CCC) -o cpu cpu_dedisp.c

tree:
	$(NVCC) -o tree dm_tree.cu $(LDFLAGS) $(INC) -L/lib
	rm -f *.cu.o *.linkinfo

subband:
	$(NVCC) -o subband subband_dedisp.cu $(LDFLAGS) $(INC) -L/lib
	rm -f *.cu.o *.linkinfo
################################################################################
