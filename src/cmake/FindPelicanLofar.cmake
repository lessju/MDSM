#
# FindPelicanLofar.cmake
#
#

find_path(PELICAN_LOFAR_INCLUDE_DIR LofarTypes.h
    PATHS
    /usr/local/pelican-lofar/include
    /usr/local/pelican-lofar
    /usr/include/
)

SET(PELICAN_LOFAR_NAMES pelican-lofar)

FOREACH(lib ${PELICAN_LOFAR_NAMES} )
    FIND_LIBRARY(PELICAN_LOFAR_LIBRARY_${lib}
        NAMES ${lib}
        PATHS
        /usr/local/pelican-lofar/lib
    )
    LIST(APPEND PELICAN_LOFAR_LIBRARIES ${PELICAN_LOFAR_LIBRARY_${lib}})
ENDFOREACH(lib)

# handle the QUIETLY and REQUIRED arguments and set CFITSIO_FOUND to TRUE if.
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(PELICAN_LOFAR DEFAULT_MSG PELICAN_LOFAR_LIBRARIES PELICAN_LOFAR_INCLUDE_DIR)

IF(NOT PELICAN_LOFAR_FOUND)
    SET( PELICAN_LOFAR_LIBRARIES )
ENDIF(NOT PELICAN_LOFAR_FOUND)
