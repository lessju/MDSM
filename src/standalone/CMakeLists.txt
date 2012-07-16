include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(standalone_lib_src
    DoubleBuffer.cpp
    UdpChunker.cpp
)

add_library(standaloneMdsmLib STATIC ${standalone_lib_src} )
target_link_libraries(standaloneMdsmLib
    ${QT_QTCORE_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    mdsm-modules
)

add_executable(standaloneMdsm standaloneMain.cpp)
target_link_libraries(standaloneMdsm
    ${QT_QTCORE_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    standaloneMdsmLib
    mdsm-cuda
    mdsm
)

add_executable(dataEmulator DataEmulator.cpp)
target_link_libraries(dataEmulator
    ${QT_QTCORE_LIBRARY}
    ${QT_QTNETWORK_LIBRARY}
    mdsm
)
