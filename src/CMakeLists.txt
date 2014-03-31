# === Set the project name and specify the required CMAKE version
project(MDSM CXX C)
cmake_minimum_required(VERSION 2.4)

# === Append the local cmake directory to the module path.
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

# === Options and defaults.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif(NOT CMAKE_BUILD_TYPE)

if(NOT MDSM_INSTALL_DIR)
    set(MDSM_INSTALL_DIR "MDSM")
endif(NOT MDSM_INSTALL_DIR)

if(NOT LIBRARY_INSTALL_DIR)
    set(LIBRARY_INSTALL_DIR "${MDSM_INSTALL_DIR}/lib")
endif(NOT LIBRARY_INSTALL_DIR)

if(NOT INCLUDE_INSTALL_DIR)
    set(INCLUDE_INSTALL_DIR "${MDSM_INSTALL_DIR}/include")
endif(NOT INCLUDE_INSTALL_DIR)

if(NOT BINARY_INSTALL_DIR)
    set(BINARY_INSTALL_DIR "${MDSM_INSTALL_DIR}/bin")
endif(NOT BINARY_INSTALL_DIR)

if(NOT GPU_VERSION_ARCHITECTURE)
    set(GPU_ARCHITECTURE "sm_35")
endif(NOT GPU_VERSION_ARCHITECTURE)

# === Compiler dependancies if compiler is GCC
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-O3 -fexpensive-optimizations")
endif()

# === Find dependencies.
if(NOT CUDA_SDK_ROOT_DIR)
    set(CUDA_SDK_ROOT_DIR /usr/local/cudaSDK/C)
endif(NOT CUDA_SDK_ROOT_DIR)

find_package(CUDA 3.0 REQUIRED)
set(CUDA_PROPERGATE_HOST_FLAGS OFF)
set(CUDA_BUILD_EMULATION OFF)
if(CMAKE_BUILD_TYPE MATCHES RELEASE|[Rr]elease)
    set(CUDA_NVCC_FLAGS --gpu-architecture ${GPU_ARCHITECTURE})
    list(APPEND CUDA_NVCC_FLAGS --ptxas-options=-v)
    list(APPEND CUDA_NVCC_FLAGS -O3)
    list(APPEND CUDA_NVCC_FLAGS --use_fast_math)
    list(APPEND CUDA_NVCC_FLAGS --compiler-options;-O3)
    list(APPEND CUDA_NVCC_FLAGS --ptxas-options=-dlcm=ca)
else(CMAKE_BUILD_TYPE MATCHES RELEASE|[Rr]elease)
    set(CUDA_NVCC_FLAGS --gpu-architecture ${GPU_ARCHITECTURE})
endif(CMAKE_BUILD_TYPE MATCHES RELEASE|[Rr]elease)

include(compilerDefaults)

#find_package(Pelican REQUIRED)
find_package(Qt4 COMPONENTS QtCore QtNetwork QtXml REQUIRED)
#find_package(PelicanLofar REQUIRED)
find_package(OpenMP REQUIRED)
find_package(FFTW3 REQUIRED)
find_package(GSL REQUIRED)
find_package(Boost COMPONENTS random)
find_package(PGPLOT REQUIRED)

include_directories("${CUDA_SDK_ROOT_DIR}/common/inc")
include_directories(${CUDA_INCLUDE_DIRS})
#include_directories(${PELICAN_LOFAR_INCLUDE_DIR})
#include_directories(${PELICAN_INCLUDE_DIR})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/mdsm")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/mdsm-coherent")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/mdsm-multibeam")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/beamformer")
#include_directories("${CMAKE_CURRENT_SOURCE_DIR}/pelican-mdsm/modules")
include_directories(${QT_QTXML_INCLUDE_DIR})
include_directories(${QT_QTCORE_INCLUDE_DIR})
include_directories(${QT_QTNETWORK_INCLUDE_DIR})
include_directories(${QT_INCLUDE_DIR})
include_directories(${PGPLOT_INCLUDE_DIRS})
include_directories(${GSL_INCLUDE_DIRS} ${GSLCBLAS_INCLUDE_DIRS})

# === Recurse into sub-directories defined in modules.cmake.
include(modules.cmake)

add_executable(standalone-mdsm MDSM.cpp)
target_link_libraries(standalone-mdsm
    ${QT_QTCORE_LIBRARY}
    ${GSL_LIBRARIES} 
    ${GSLCBLAS_LIBRARIES}
    mdsm-multibeam-cuda
    mdsm-multibeam
)

set_target_properties(standalone-mdsm PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS} -std=c++0x"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "-lgsl -lgslcblas"
)
#install(TARGETS "standalone-mdsm" DESTINATION ${MDSM_INSTALL_DIR})

add_executable(standalone-beamformer beamformer.cpp)
target_link_libraries(standalone-beamformer
    ${QT_QTCORE_LIBRARY}
    beamformer-cuda
    beamformer
)
#install(TARGETS "standalone-beamformer" DESTINATION ${MDSM_INSTALL_DIR})

