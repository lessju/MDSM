###############################################################################

BIN = /home/lessju/sigproc/cuda_dedispersion
CCC = g++

###############################################################################

CC = $(CCC) -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64

NVCC 		:= nvcc
CUDA_PATH     	:= /usr/local/cuda
CUDA_SDK_PATH 	:= /usr/local/cuda_sdk/C
CUDA_INC	:= -I. -I${CUDA_SDK_PATH}/common/inc -I${CUDA_PATH}/include

ALL_INC		:= -I. -I${CUDA_SDK_PATH}/common/inc -I${CUDA_PATH}/include \
                   -I/usr/include/qt4/QtCore -I/usr/include/qt4/QtGui -I/usr/include/qt4/QtXml \
                   -I/usr/include/qt4/QtSql -I/usr/include/qt4/QtNetwork -I/usr/include/qt4 \
                   -I/usr/local/pelican-lofar/include

LDFLAGS 	:= -L${CUDA_PATH}/lib -L${CUDA_PATH}/lib64 -L${CUDA_SDK_PATH}/lib -L. \
 	           -L${CUDA_SDK_PATH}/common/lib -L/usr/local/pelican-lofar/lib -L/lib \
         	   -lcuda -lcudart -lcutil \
               -lQtGui -lQtCore -lQtSql -lQtNetwork -lQtXml \
               -lm -lpthread -lpelican -lpelican-lofar

FILES		:= MDSM.cpp dedispersion_manager.cpp dedispersion_output.cpp \
               file_handler.cpp PelicanLofarClient.cpp dedispersion_wrapper.o
	
############################ CUDA DEDISPERSE ###################################

#FIXME: Had to copy libpelican-lofar to /usr/lib

MDSM : 	dedisperse_wrapper.o
	$(CCC) -o MDSM $(FILES) $(ALL_INC) $(LDFLAGS) -Wno-write-strings
	rm -f *.cu.o *.linkinfo *.o

dedisperse_wrapper.o:
	$(NVCC) -c dedispersion_wrapper.cu --gpu-architecture sm_13 $(CUDA_INC)

################################################################################
