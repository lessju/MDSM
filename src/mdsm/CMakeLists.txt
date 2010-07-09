# Cuda source files
set(nvcc_SRC
    dedispersion_kernel.cu
    dedispersion_wrapper.cu
    dedispersion_thread.cu
)

CUDA_ADD_LIBRARY(mdsm-cuda STATIC ${nvcc_SRC})

include_directories(${CUDA_INCLUDE_DIRS})

target_link_libraries(mdsm-cuda
    ${CUDA_LIBRARIES}
)

# C++ source files
set(dedispersion_SRC
    dedispersion_manager.cpp
    dedispersion_output.cpp
)

add_library(mdsm STATIC ${dedispersion_SRC})

target_link_libraries(mdsm
    ${QT_QTCORE_LIBRARY}
    mdsm-cuda
)
 
set_target_properties(mdsm PROPERTIES
    LINK_FLAGS "-lpthread"
)


# === Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()


# Recurse into test subdirectory.
#add_subdirectory(test)
