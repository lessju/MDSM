include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(pipeline_lib_src
    src/MdsmPipeline.cpp
    src/SigprocMdsmPipeline.cpp
    src/AtaMdsmPipeline.cpp
    src/DDStream1.cpp
    src/DDStream2.cpp
)

add_library(pelicanMdsm ${pipeline_lib_src} )
target_link_libraries(pelicanMdsm
    ${PELICAN_LOFAR_LIBRARIES}
    ${PELICAN_LIBRARY}
    ${FFTW3_LIBRARIES}
    mdsm-modules
)

add_executable(mdsmPipeline src/main.cpp)

# Link with openmp.
set_target_properties(mdsmPipeline PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)

target_link_libraries(mdsmPipeline
    pelicanMdsm
)

add_executable(DDStream1 src/mainDDStream1.cpp)

# Link with openmp.
set_target_properties(mdsmPipeline PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)

target_link_libraries(DDStream1
    pelicanMdsm
)
add_executable(DDStream2 src/mainDDStream2.cpp)

# Link with openmp.
set_target_properties(mdsmPipeline PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)

target_link_libraries(DDStream2
    pelicanMdsm
)

add_executable(sigprocPipeline src/sigprocMain.cpp)
target_link_libraries(sigprocPipeline
    pelicanMdsm
)

add_executable(ataPipeline src/ataMain.cpp)

# Link with openmp.
set_target_properties(ataPipeline PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)

target_link_libraries(ataPipeline
    pelicanMdsm
)

install(TARGETS mdsmPipeline DESTINATION ${BINARY_INSTALL_DIR})

include(CopyFiles)
copy_files(${CMAKE_CURRENT_SOURCE_DIR}/data/*.xml . mdsmXmlFiles)
copy_files(${CMAKE_CURRENT_SOURCE_DIR}/data/*.dat . channeliserDatFiles)


#add_subdirectory(test)
