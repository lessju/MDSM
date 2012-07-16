###############################################################################

BIN = /home/lessju/sigproc/cuda_dedispersion
CCC = g++

###############################################################################

CC = $(CCC) $(DFITS) $(DFFTW) -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64

NVCC 		:= nvcc
CUDA_PATH     	:= /usr/local/cuda
CUDA_SDK_PATH 	:= /usr/local/cudaSDK/C
PRESO   := /home/lessju/Code/presto
INC		:= -I. -I${CUDA_SDK_PATH}/common/inc -I${CUDA_PATH}/include
LDFLAGS 	:= -L${CUDA_PATH}/lib -L${CUDA_PATH}/lib64 -L${CUDA_SDK_PATH}/lib \
	          -L${CUDA_SDK_PATH}/common/lib -L/lib \
        	  -lcuda -lcudart  -lm
	
############################ CUDA DEDISPERSE ###################################

binner:
	$(NVCC) -o binner binner.cu $(LDFLAGS) $(INC) -L/lib
	rm -f *.cu.o *.linkinfo

detrend:
	$(NVCC) -o detrend detrend.cu $(LDFLAGS) $(INC) -lfftw3
	rm -f *.cu.o *.linkinfo

singlepulse:
	g++ -o singlepulse singlepulse.cpp -lfftw3f -fopenmp -g \
      -msse4.1 -O3 -ffast-math -fno-exceptions -Wall -ftree-vectorize -fexpensive-optimizations

singlepulse_presto:
	g++ -o singlepulse_presto singlepulse_presto.cpp -I$(PRESTO)/include -L. -lpresto -fopenmp \
     -msse4.1 -O3 -ffast-math -fno-exceptions -Wall -ftree-vectorize -fexpensive-optimizations
fft:
	$(NVCC) -o fft cuda_fft.cu $(LDFLAGS) $(INC) -L/lib -lcufft
	rm -f *.cu.o *.linkinfo

cpu_dedisp:
	$(CCC) -o cpu cpu_dedisp.c

tree:
	$(NVCC) -o tree dm_tree.cu $(LDFLAGS) $(INC) -L/lib
	rm -f *.cu.o *.linkinfo

subband:
	$(NVCC) -o subband subband_dedisp.cu --gpu-architecture sm_20 $(LDFLAGS) $(INC) -L/lib
	rm -f *.cu.o *.linkinfo

average:
	$(NVCC) -o average average.cu $(LDFLAGS) $(INC) -L/lib
	rm -f *.cu.o *.linkinfo

brute_dedisp:
	$(NVCC) -o brute_dedisp brute_dedisp.cu --ptxas-options=-v $(LDFLAGS) $(INC) -L/lib
	rm -f *.cu.o *.linkinfo
	
brute_dedisp_imp:
	$(NVCC) -o brute_dedisp_imp brute_dedisp_imp.cu $(LDFLAGS) $(INC) -L/lib
	rm -f *.cu.o *.linkinfo

test_atomic:
	$(NVCC) -o test_atomic testAtomicCas.cu -arch compute_20 $(LDFLAGS) $(INC) -L/lib
	rm -f *.cu.o *.linkinfo

coherent_dedisp:
#	$(NVCC) -c coherent_dedisp.cu $(INC) --use_fast_math --gpu-architecture sm_13 -I/usr/local/pgplot/
	$(NVCC) -c coherent_dedisp.cu $(INC) --use_fast_math --gpu-architecture sm_13 -I/usr/local/pgplot/  
	gfortran -o coherent coherent_dedisp.o $(LDFLAGS) -L/lib -lcufft -L/usr/local/pgplot -lcpgplot -lpgplot -lgfortran -L/usr/X11R6/lib -lX11
	rm -f *.cu.o *.linkinfo


fft_shift:
	$(NVCC) -c fftShift.cu $(INC) --use_fast_math --gpu-architecture sm_13 -I/usr/local/pgplot/  
	gfortran -o fftShift fftShift.o $(LDFLAGS) -L/lib -lcufft -L/usr/local/pgplot -lcpgplot -lpgplot -lgfortran -L/usr/X11R6/lib -lX11
	rm -f *.cu.o *.linkinfo

################################################################################
