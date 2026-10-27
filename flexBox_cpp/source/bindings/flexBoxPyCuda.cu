// CUDA build of the Python bindings: same source as flexBoxPy.cpp, compiled by nvcc
// so that __CUDACC__ is defined and the GPU solver / thrust::device_vector data are used.
#define FLEXBOX_MODULE_NAME flexBoxPyCuda
#include "flexBoxPy.cpp"
