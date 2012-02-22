set(mdsm_modules_SRC
    MdsmModule.cpp
    DedispersedDataWriter.cpp
    RawVoltageAdapter.cpp
    SigprocAdapter.cpp
    GuppiAdapter.cpp
    SpeadBeamChunker.cpp
    SpeadBeamAdapterTimeSeries.cpp
    SpeadBeamDataClient.cpp
    HeapBuffer.cpp
)

add_library(mdsm-modules ${mdsm_modules_SRC})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}../../mdsm")
target_link_libraries(mdsm-modules
    ${PELICAN_LOFAR_LIBRARIES}
    ${PELICAN_LIBRARIES}
    mdsm
)

