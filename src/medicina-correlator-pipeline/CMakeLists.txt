include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(standalone_lib_src
    DoubleBuffer.cpp
    PacketChunker.cpp
)

add_library(standaloneCorrelatorLib STATIC ${standalone_lib_src} )
target_link_libraries(standaloneCorrelatorLib
    ${QT_QTCORE_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
)

add_executable(medicinaCorrelatorPipeline pipeline.cpp)
target_link_libraries(medicinaCorrelatorPipeline
    ${QT_QTCORE_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    standaloneCorrelatorLib
    mdsm-multibeam-cuda
    mdsm-multibeam
)

set_target_properties(medicinaCorrelatorPipeline PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
