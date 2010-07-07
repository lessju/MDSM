#
#

# Set the C++ release flags.
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DQT_NO_DEBUG -DNDEBUG")

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-Wall -Wextra)
    add_definitions(-Wno-deprecated -Wno-unknown-pragmas -Wno-write-strings)
    list(APPEND CPP_PLATFORM_LIBS util dl)
elseif(CMAKE_CXX_COMPILER MATCHES icpc)
    add_definitions(-Wall -Wcheck)
    add_definitions(-wd383 -wd981)  # Suppress remarks / warnings.
    add_definitions(-ww111 -ww1572) # Promote remarks to warnings.
else(CMAKE_COMPILER_IS_GNUCXX)
    # use defaults (and pray it works...)
endif(CMAKE_COMPILER_IS_GNUCXX)

if(APPLE)
    add_definitions(-DDARWIN)
endif(APPLE)




