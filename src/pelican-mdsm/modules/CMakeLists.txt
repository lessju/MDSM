set(mdsm_modules_SRC
    MDSMModule.cpp
)

add_library(mdsm-modules ${mdsm_modules_SRC})

target_link_libraries(
    mdsm-modules
    ${PELICAN_LIBRARIES}
    ${PELICAN_LOFAR_LIBRARIES}
)

