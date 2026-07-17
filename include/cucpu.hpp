#pragma once

#include <cstdlib>
#include <cstring>

namespace cucpu {

//! dim3 (grid/block dimension type)
struct dim3 {
    unsigned x, y, z;

    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1): x(x_), y(y_), z(z_) {}
};


// built in variables:
//  this emulation uses thread local globals that will be set before each thread runs
inline thread_local dim3 threadIdx;
inline thread_local dim3 blockIdx;
inline thread_local dim3 blockDim;
inline thread_local dim3 gridDim;

//! host / device memory primitives
/**
    Since this is designed to purely run on CPU, there is only one address
   space:
    * "device" memory -> malloc
    * cudaMemCpy -> memcpy
    etc.
 */
enum cudaMemCpyKind {
  cudaMemcpyHostToDevice,
  cudaMemcpyToDeviceHost,
  cudaMemcpyDeviceToDevice,
  cudaMemcpyHostToHost,
};

using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaMallocFail = 1;

cudaError_t cudaMalloc(void **p, size_t sz) {
  *p = std::malloc(sz);
  return *p ? cudaSuccess : cudaMallocFail;
}

cudaError_t cudaFree(void *p) {
  if (p)
    std::free(p);
  return cudaSuccess;
}

cudaError_t cudaMemcpy(void *dst, const void* src, size_t n, cudaMemCpyKind _mode) {
    std::memcpy(dst,src, n);
    return cudaSuccess;
}

cudaError_t cudaMemset(void *p, int v, size_t n) {
    std::memset(p, v, n);
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize() {
    return cudaSuccess;
}
} // namespace cucpu