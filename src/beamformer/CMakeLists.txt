# Cuda source files
set(nvcc_beamformer_SRC
    beamforming_kernel.cu
    beamforming_wrapper.cu
    beamforming_thread.cu
)

CUDA_ADD_LIBRARY(beamformer-cuda STATIC ${nvcc_beamformer_SRC})

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

target_link_libraries(beamformer-cuda
    ${CUDA_LIBRARIES}
    ${OpenMP_CXX_FLAGS}
    ${CUDA_CUFFT_LIBRARIES}
)

# C++ source files
set(beamformer_SRC
    beamforming_manager.cpp
    beamforming_output.cpp
    beamshifts.cpp
)

add_library(beamformer STATIC ${beamformer_SRC})

target_link_libraries(beamformer
    ${QT_QTCORE_LIBRARY}
    ${QT_QTXML_LIBRARY}
    ${GSL_LIBRARIES} 
    ${GSLCBLAS_LIBRARIES}
    ${PGPLOT_LIBRARIES}
    beamformer-cuda
)
 
set_target_properties(beamformer PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "-lpthread"
)

# === Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()

