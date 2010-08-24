set(standalone_SRC
    file_handler.cpp
)

add_library(standalone STATIC ${standalone_SRC})

target_link_libraries(standalone
)

