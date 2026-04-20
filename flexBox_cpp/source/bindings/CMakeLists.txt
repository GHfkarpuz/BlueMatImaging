find_package(pybind11 REQUIRED)

pybind11_add_module(flexBoxPy flexBoxPy.cpp)

# =========================
# Link against Core Library
# =========================
target_link_libraries(flexBoxPy PRIVATE flexBoxLib)

# =========================
# Include Paths
# =========================
target_include_directories(flexBoxPy PRIVATE
    ${CMAKE_SOURCE_DIR}/source
)

install(TARGETS flexBoxPy
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)