# Cuda source files
set(nvcc_SRC
    coherent_dedispersion_kernel.cu
    coherent_dedispersion_wrapper.cu
    coherent_dedispersion_thread.cu
)

CUDA_ADD_LIBRARY(mdsm-coherent-cuda STATIC ${nvcc_SRC})

include_directories(${CUDA_INCLUDE_DIRS})

target_link_libraries(mdsm-coherent-cuda
    ${CUDA_LIBRARIES}
    ${CUDA_CUFFT_LIBRARIES}
)

# C++ source files
set(dedispersion_SRC
    coherent_dedispersion_manager.cpp
    coherent_dedispersion_output.cpp
)

add_library(mdsmCoherent STATIC ${dedispersion_SRC})

target_link_libraries(mdsmCoherent
    ${QT_QTCORE_LIBRARY}
    ${QT_QTXML_LIBRARY}
    ${PGPLOT_LIBRARIES}
    mdsm-coherent-cuda
)
 
set_target_properties(mdsmCoherent PROPERTIES
    LINK_FLAGS "-lpthread"
)

# ------------- Create Test Executable ------------------
add_executable(mdsmCoherentExe MDSM.cpp)

target_link_libraries(mdsmCoherentExe
    mdsmCoherent
    ${PGPLOT_LIBRARIES}
)

# === Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()
