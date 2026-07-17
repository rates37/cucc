#pragma once

#include <cstdlib>
#include <cstring>

namespace cucpu {

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