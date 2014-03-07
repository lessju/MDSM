include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(standalone_lib_src
    DoubleBuffer.cpp
    PacketChunker.cpp
)

add_library(standaloneBeamformerLib STATIC ${standalone_lib_src} )
target_link_libraries(standaloneBeamformerLib
    ${QT_QTCORE_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
)

add_executable(medicinaBeamformer pipeline.cpp)
target_link_libraries(medicinaBeamformer
    ${QT_QTCORE_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    standaloneBeamformerLib
    beamformer-cuda
    beamformer
)

set_target_properties(medicinaBeamformer PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
install(TARGETS "medicinaBeamformer" DESTINATION ${MDSM_INSTALL_DIR})
