# === Set the project name and specify the required CMAKE version
project(MDSM CXX C)
cmake_minimum_required(VERSION 2.4)

# === Append the local cmake directory to the module path.
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

# === Options and defaults.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE debug)
endif(NOT CMAKE_BUILD_TYPE)

if(NOT MDSM_INSTALL_DIR)
    set(MDSM_INSTALL_DIR "mdsm")
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
    set(GPU_ARCHITECTURE "sm_13")
endif(NOT GPU_VERSION_ARCHITECTURE)


# === Find dependencies.
if(NOT CUDA_SDK_ROOT_DIR)
    set(CUDA_SDK_ROOT_DIR /usr/local/cudaSDK/C)
endif(NOT CUDA_SDK_ROOT_DIR)

find_package(CUDA 2.3 REQUIRED)
set(CUDA_PROPERGATE_HOST_FLAGS OFF)
set(CUDA_BUILD_EMULATION OFF)
if(CMAKE_BUILD_TYPE MATCHES RELEASE|[Rr]elease)
    set(CUDA_NVCC_FLAGS --gpu-architecture ${GPU_ARCHITECTURE})
else(CMAKE_BUILD_TYPE MATCHES RELEASE|[Rr]elease)
    set(CUDA_NVCC_FLAGS --gpu-architecture sm_20 ${GPU_ARCHITECTURE})
endif(CMAKE_BUILD_TYPE MATCHES RELEASE|[Rr]elease)


include(compilerDefaults)


find_package(Pelican REQUIRED)
find_package(Qt4 COMPONENTS QtCore QtNetwork QtXml REQUIRED)
find_package(PelicanLofar REQUIRED)
find_package(OpenMP REQUIRED)
find_package(FFTW3 REQUIRED)
# pthread ... ?!


include_directories("${CUDA_SDK_ROOT_DIR}/common/inc")
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PELICAN_LOFAR_INCLUDE_DIR})
include_directories(${PELICAN_INCLUDE_DIR})

# === Recurse into sub-directories defined in modules.cmake.
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/mdsm")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/standalone")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/pelican-mdsm/modules")
include_directories(${QT_QTXML_INCLUDE_DIR})
include(modules.cmake)


add_executable(standalone-mdsm MDSM.cpp)
target_link_libraries(standalone-mdsm
    ${QT_QTCORE_LIBRARY}
    ${PELICAN_LOFAR_LIBRARIES}
    ${PELICAN_LIBRARIES}
    standalone
    mdsm-cuda
    mdsm
)
