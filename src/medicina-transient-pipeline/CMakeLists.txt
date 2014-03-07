include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(standalone_lib_src
    DoubleBuffer.cpp
    SpeadBeamChunker.cpp
)

add_library(standaloneMdsmLib STATIC ${standalone_lib_src} )
target_link_libraries(standaloneMdsmLib
    ${QT_QTCORE_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
)

add_executable(medicinaTransientPipeline pipeline.cpp)
target_link_libraries(medicinaTransientPipeline
    ${QT_QTCORE_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    standaloneMdsmLib
    mdsm-multibeam-cuda
    mdsm-multibeam
)

set_target_properties(medicinaTransientPipeline PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
install(TARGETS "medicinaTransientPipeline" DESTINATION ${MDSM_INSTALL_DIR})
