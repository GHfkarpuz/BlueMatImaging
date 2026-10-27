find_package(pybind11 REQUIRED)

pybind11_add_module(flexBoxPy flexBoxPy.cpp)

# =========================
# Link against Core Library
# =========================
target_link_libraries(flexBoxPy PRIVATE flexBoxLib)

if(USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(flexBoxPy PRIVATE OpenMP::OpenMP_CXX)
endif()

# =========================
# CUDA version (GPU)
# =========================
if(FLEXBOX_BUILD_CUDA)
    pybind11_add_module(flexBoxPyCuda flexBoxPyCuda.cu)
    target_link_libraries(flexBoxPyCuda PRIVATE flexBoxLib)
    set_target_properties(flexBoxPyCuda PROPERTIES CUDA_ARCHITECTURES "${FLEXBOX_CUDA_ARCHITECTURES}")
    target_compile_options(flexBoxPyCuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)

    if(USE_OPENMP AND OpenMP_CXX_FOUND)
        target_compile_options(flexBoxPyCuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
        target_link_libraries(flexBoxPyCuda PRIVATE OpenMP::OpenMP_CXX)
    endif()

    install(TARGETS flexBoxPyCuda
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    )
endif()

# =========================
# Include Paths
# =========================
target_include_directories(flexBoxPy PRIVATE
    ${CMAKE_SOURCE_DIR}/source
)

install(TARGETS flexBoxPy
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)