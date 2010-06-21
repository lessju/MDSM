###############################################################################

BIN = /home/lessju/sigproc/cuda_dedispersion
CCC = g++

###############################################################################

CC = $(CCC) $(DFITS) $(DFFTW) -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64

NVCC 		:= nvcc
CUDA_PATH     	:= /usr/local/cuda
CUDA_SDK_PATH 	:= /usr/local/cuda_sdk/C
INC		:= -I. -I${CUDA_SDK_PATH}/common/inc -I${CUDA_PATH}/include \
                   -I/usr/include/qt4/QtCore -I/usr/include/qt4/QtGui -I/usr/include/qt4
LDFLAGS 	:= -L${CUDA_PATH}/lib -L${CUDA_PATH}/lib64 -L${CUDA_SDK_PATH}/lib \
	          -L${CUDA_SDK_PATH}/common/lib -L/lib \
        	  -lcuda -lcudart -lcutil \
                  -lm -lpthread -lpelican -lQtGui -lQtCore
	
############################ CUDA DEDISPERSE ###################################

cudadedisperse :
	$(NVCC) -c dedispersion_manager.cu --gpu-architecture sm_13 $(INC)
	g++ -o MDSM MDSM.cpp  dedispersion_manager.o -Wno-write-strings $(LDFLAGS) $(INC) $(LDFLAGS) 
	rm -f *.cu.o *.linkinfo *.o

binner:
	$(NVCC) -o binner binner.cu $(LDFLAGS) -L/lib
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
