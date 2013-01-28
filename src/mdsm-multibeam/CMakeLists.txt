# Cuda source files
set(nvcc_multibeam_SRC
    multibeam_dedispersion_kernel.cu
    multibeam_dedispersion_wrapper.cu
    multibeam_dedispersion_thread.cu
)

CUDA_ADD_LIBRARY(mdsm-multibeam-cuda STATIC ${nvcc_multibeam_SRC})

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

target_link_libraries(mdsm-multibeam-cuda
    ${CUDA_LIBRARIES}
    ${OpenMP_CXX_FLAGS}
)

# C++ source files
set(dedispersion_multibeam_SRC
    multibeam_dedispersion_manager.cpp
    multibeam_dedispersion_output.cpp
    multibeam_dedispersion_writer.cpp
    file_handler.cpp
    dbscan.cpp
)

add_library(mdsm-multibeam STATIC ${dedispersion_multibeam_SRC})

target_link_libraries(mdsm-multibeam
    ${QT_QTCORE_LIBRARY}
    ${QT_QTXML_LIBRARY}
    ${GSL_LIBRARIES} 
    ${GSLCBLAS_LIBRARIES}
    ${PGPLOT_LIBRARIES}
    mdsm-multibeam-cuda
)
 
set_target_properties(mdsm-multibeam PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "-lpthread"
)

# === Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()

