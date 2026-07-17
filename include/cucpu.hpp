#pragma once

#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace cucpu {

//! dim3 (grid/block dimension type)
struct dim3 {
  unsigned x, y, z;

  dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1)
      : x(x_), y(y_), z(z_) {}
};

// built in variables:
//  this emulation uses thread local globals that will be set before each thread
//  runs
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

inline cudaError_t cudaMalloc(void **p, size_t sz) {
  *p = std::malloc(sz);
  return *p ? cudaSuccess : cudaMallocFail;
}

inline cudaError_t cudaFree(void *p) {
  if (p)
    std::free(p);
  return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void *dst, const void *src, size_t n,
                              cudaMemCpyKind _mode) {
  std::memcpy(dst, src, n);
  return cudaSuccess;
}

inline cudaError_t cudaMemset(void *p, int v, size_t n) {
  std::memset(p, v, n);
  return cudaSuccess;
}

inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }

//! Barrier (for synchronisation)
// Allows a fixed number of threads to synchronise repeatedly
// without risking race conditions between consecutive cycles
class Barrier {
public:
  Barrier(unsigned int total_threads)
      : total_threads_(total_threads), threads_remaining_(total_threads),
        generation_(0) {}

  // blocks the calling thread until all expected threads arrive
  void wait() {
    std::unique_lock<std::mutex> lock(
        mutex_); // local lock, gets destroyed upon method return/exit by
                 // destructor (RAII)

    // get current generation
    // this lets the local thread know when a new cycle has begun
    unsigned int current_generation = generation_;

    // the final thread to finish triggers the transition:
    if (--threads_remaining_ == 0) {
      // advance to next generation:
      generation_++;

      // reset counter for next generation:
      threads_remaining_ = total_threads_;

      // signal all other threads:
      cv_.notify_all();
    } else {
      // wait until last thread notifies:
      cv_.wait(lock, [this, current_generation] {
        return current_generation != generation_;
      });
    }
  }

private:
  const unsigned int total_threads_;
  unsigned int threads_remaining_;
  unsigned int generation_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

//! Per-block context: Barrier + shared block memory
struct BlockContext {
  Barrier *barrier = nullptr;
  // guard access to shared memory
  std::mutex shared_mem_mutex;
  // map unique ID to memory addr
  std::unordered_map<int, void *> shared_mem;
  // list of cleanup functions to properly call destructors
  std::vector<std::function<void()>> deleters;

  ~BlockContext() {
    // ensure all allocated shared memory is correctly cleaned
    for (auto &d : deleters)
      d();
  }
};

inline thread_local BlockContext *current_thread_block = nullptr;

// Variables that are __shared__ resolve to a per-block object, allocated once
// by whichever thread reaches the declaration first, and shared by all of the
// threads in the block. The `id` is a unique tag the transpiler should assign
template <class T> T &shared_var(int id) {
  BlockContext *bc = current_thread_block;
  std::lock_guard<std::mutex> lock(bc->shared_mem_mutex);

  // Check if the variable has already been allocated by another thread
  auto it = bc->shared_mem.find(id);
  if (it != bc->shared_mem.end())
    return *static_cast<T *>(it->second);

  // allocate the variable
  T *p = new T();
  bc->shared_mem.emplace(id, static_cast<void *>(p));

  // capture the pointer so it can be correctly deleted during the BlockContext
  // destructor invocation
  bc->deleters.push_back([p] { delete p; });
  return *p;
}

// block synchronisation barrier, emulates CUDA's `__syncthreads()`
// Blocks all threads in the active block until they have all arrived at this
// point.
inline void syncthreads() {
  if (current_thread_block && current_thread_block->barrier) {
    current_thread_block->barrier->wait();
  }
}

} // namespace cucpu