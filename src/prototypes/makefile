###############################################################################

CCC = g++

###############################################################################

CC = $(CCC) $(DFITS) $(DFFTW) -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64

NVCC 		:= nvcc
CUDA_PATH     	:= /usr/local/cuda-5.5
CUDA_SDK_PATH 	:= /usr/local/cudaSDK/C
PRESTO   := /home/lessju/Code/presto
INC		:= -I. -I${CUDA_SDK_PATH}/common/inc -I${CUDA_PATH}/include
LDFLAGS 	:= -L${CUDA_PATH}/lib -L${CUDA_PATH}/lib64 -L${CUDA_SDK_PATH}/lib \
	          -L${CUDA_SDK_PATH}/common/lib -L/lib \
        	  -lcuda -lcudart  -lm

############################ CUDA DEDISPERSE ###################################

binner:
	$(NVCC) -o binner binner.cu $(LDFLAGS) $(INC) -L/lib
	rm -f *.cu.o *.linkinfo

detrend:
	$(NVCC) -o detrend detrend.cu --gpu-architecture sm_20 $(LDFLAGS) $(INC)
	rm -f *.cu.o *.linkinfo

median_filter:
	$(NVCC) -o median_filter median_filter.cu $(LDFLAGS) $(INC) --ptxas-options=-v --gpu-architecture=sm_20 --use_fast_math
	rm -f *.cu.o *.linkinfo

mean_stddev:
	$(NVCC) -o mean_stddev mean_stddev.cu $(LDFLAGS) $(INC) --ptxas-options=-v --gpu-architecture=sm_20 --use_fast_math
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
	$(NVCC) -o brute_dedisp_imp brute_dedisp_imp.cu $(LDFLAGS) $(INC) -L/lib --use_fast_math --gpu-architecture=sm_20
	rm -f *.cu.o *.linkinfo

test_atomic:
	$(NVCC) -o test_atomic testAtomicCas.cu -arch compute_20 $(LDFLAGS) $(INC) -L/lib
	rm -f *.cu.o *.linkinfo

coherent_dedisp:
	$(NVCC) -c coherent_dedisp.cu $(INC) --use_fast_math --gpu-architecture sm_13 -I/usr/local/pgplot/  
	gfortran -o coherent coherent_dedisp.o $(LDFLAGS) -L/lib -lcufft -L/usr/local/pgplot -lcpgplot -lpgplot -lgfortran -L/usr/X11R6/lib -lX11
	rm -f *.cu.o *.linkinfo

huber_filter:
	g++ -o huber_filter huber_filter.cpp -O3 -fexpensive-optimizations -lboost_random -fopenmp
	rm -f *.o

quantiser:
	g++ -o quantiser data_quantisation.cpp file_handler.cpp -O3 -fexpensive-optimizations -lboost_random -fopenmp
	rm -f *.o

bandpass:
	$(CCC) -c file_handler.cpp
	$(NVCC) -o bandpass bandpass.cu file_handler.o $(INC) $(LDFLAGS) --gpu-architecture sm_20 --use_fast_math -lgsl -lgslcblas --ptxas-options=-v
	rm -f *.cu.o *.linkinfo

beamformer:
	$(NVCC) -o beamformer beamformer.cu $(INC) $(LDFLAGS) --gpu-architecture sm_35 --use_fast_math --ptxas-options=-v 
	rm -f *.cu.o *.linkinfo

beamformer_medicina:
	$(NVCC) -o beamformer_medicina beamformer_medicina.cu $(INC) $(LDFLAGS) --gpu-architecture sm_35 --use_fast_math --ptxas-options=-v 
	rm -f *.cu.o *.linkinfo

periodicity:
	$(CCC) -c file_handler.cpp
	$(NVCC) -o periodicity periodicity_search.cu file_handler.o $(INC) $(LDFLAGS) --gpu-architecture sm_20 --use_fast_math --ptxas-options=-v -lcufft
	rm -f *.cu.o *.linkinfo

imager:
	$(NVCC) -o imagere imager.cu $(INC) $(LDFLAGS) --gpu-architecture sm_20 --ptxas-options=-v -O3 -ptx
	rm -f *.cu.o *.linkinfo

power:
	$(NVCC) -o power power.cu $(INC) $(LDFLAGS) --gpu-architecture sm_20 --ptxas-options=-v -O3
	rm -f *.cu.o *.linkinfo

fft_shift:
	$(NVCC) -c fftShift.cu $(INC) --use_fast_math --gpu-architecture sm_13 -I/usr/local/pgplot/  
	gfortran -o fftShift fftShift.o $(LDFLAGS) -L/lib -lcufft -L/usr/local/pgplot -lcpgplot -lpgplot -lgfortran -L/usr/X11R6/lib -lX11
	rm -f *.cu.o *.linkinfo

heap_test:
	$(CCC) -o heap_test heap_memory_copy_sse.c -O2
	rm -f *.o

################################################################################
