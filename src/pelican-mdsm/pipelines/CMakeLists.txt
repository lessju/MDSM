include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(pipeline_lib_src
    src/MdsmPipeline.cpp
    src/GuppiMdsmPipeline.cpp
    src/SigprocMdsmPipeline.cpp
    src/DDStream1.cpp
    src/DDStream2.cpp
    src/MedicinaMdsmPipeline.cpp
    src/MedicinaCoherentPipeline.cpp
    src/CoherentTestPipeline.cpp
)

# ------------- Create pipeline library ------------------

add_library(pelicanMdsm ${pipeline_lib_src} )
target_link_libraries(pelicanMdsm
    ${PELICAN_LOFAR_LIBRARIES}
    ${PELICAN_LIBRARY}
    ${FFTW3_LIBRARIES}
    mdsm-modules
)

# ------------- Create mdsm pipeline executable ------------------
add_executable(mdsmPipeline src/main.cpp)

# Link with openmp.
set_target_properties(mdsmPipeline PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)

target_link_libraries(mdsmPipeline
    pelicanMdsm
)

# ------------- Create DDStream1 executable ------------------
add_executable(DDStream1 src/mainDDStream1.cpp)

# Link with openmp.
set_target_properties(mdsmPipeline PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)

target_link_libraries(DDStream1
    pelicanMdsm
)

# ------------- Create DDStream2 executable ------------------
add_executable(DDStream2 src/mainDDStream2.cpp)

# Link with openmp.
set_target_properties(mdsmPipeline PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)

target_link_libraries(DDStream2
    pelicanMdsm
)

# ------------- Create Sigproc pipeline executable ------------------
add_executable(sigprocPipeline src/sigprocMain.cpp)
target_link_libraries(sigprocPipeline
    pelicanMdsm
)

# ------------- Create Guppi pipeline executable ------------------
add_executable(guppiPipeline src/guppiMain.cpp)

# Link with openmp.
set_target_properties(guppiPipeline PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)

target_link_libraries(guppiPipeline
    pelicanMdsm
)

# ------------- Create Medicina pipeline executable ------------------
add_executable(medicinaPipeline src/medicinaMain.cpp)

# Link with openmp.
set_target_properties(medicinaPipeline PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)

target_link_libraries(medicinaPipeline
    pelicanMdsm
)

# ------------- Create Medicina coherent pipeline executable ------------------
add_executable(medicinaCoherentPipeline src/coherentMedicinaMain.cpp)

target_link_libraries(medicinaCoherentPipeline
    pelicanMdsm
)

# ------------- Create Test pipeline executable ------------------
add_executable(coherentTestPipeline src/coherentTestMain.cpp)

target_link_libraries(coherentTestPipeline
    pelicanMdsm
)

# ------------- Add install paths  ------------------

install(TARGETS mdsmPipeline DESTINATION ${BINARY_INSTALL_DIR})
install(TARGETS DDStream1 DESTINATION ${BINARY_INSTALL_DIR})
install(TARGETS DDStream2 DESTINATION ${BINARY_INSTALL_DIR})

include(CopyFiles)
