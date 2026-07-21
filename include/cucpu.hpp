#pragma once

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <new>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#define MINICORO_IMPL
#include "minicoro.h"

namespace cucpu {

//! dim3 (grid/block dimension type)
struct dim3 {
  unsigned x, y, z;

  dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1)
      : x(x_), y(y_), z(z_) {}
};

// index helpers:
inline dim3 lin_to_dim(unsigned int id, dim3 d) {
  dim3 r;
  r.x = id % d.x;
  r.y = (id / d.x) % d.y;
  r.z = id / (d.x * d.y);
  return r;
}

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
  cudaMemcpyDeviceToHost,
  cudaMemcpyDeviceToDevice,
  cudaMemcpyHostToHost,
};

//! CUDA-aligned error codes:
enum cudaError_t {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInitializationError = 3,
  cudaErrorInvalidConfiguration = 9,
  cudaErrorInvalidDeviceFunction = 98,
  cudaErrorLaunchFailure = 719,
  cudaErrorUnknown = 999
};

namespace detail {
inline std::atomic<cudaError_t> g_last_error{cudaSuccess};
inline void set_error(cudaError_t e) {
  if (e != cudaSuccess)
    g_last_error.store(e, std::memory_order_relaxed);
}
} // namespace detail

inline cudaError_t cudaGetLastError() {
  return detail::g_last_error.exchange(cudaSuccess, std::memory_order_relaxed);
}

inline cudaError_t cudaPeekAtLastError() {
  return detail::g_last_error.load(std::memory_order_relaxed);
}

inline const char *cudaGetErrorName(cudaError_t e) {
  switch (e) {
  case cudaSuccess:
    return "cudaSuccess";
  case cudaErrorInvalidValue:
    return "cudaErrorInvalidValue";
  case cudaErrorMemoryAllocation:
    return "cudaErrorMemoryAllocation";
  case cudaErrorInitializationError:
    return "cudaErrorInitializationError";
  case cudaErrorInvalidConfiguration:
    return "cudaErrorInvalidConfiguration";
  case cudaErrorInvalidDeviceFunction:
    return "cudaErrorInvalidDeviceFunction";
  case cudaErrorLaunchFailure:
    return "cudaErrorLaunchFailure";
  default:
    return "cudaErrorUnknown";
  }
}

inline const char *cudaGetErrorString(cudaError_t e) {
  switch (e) {
  case cudaSuccess:
    return "no error";
  case cudaErrorInvalidValue:
    return "invalid argument";
  case cudaErrorMemoryAllocation:
    return "out of memory";
  case cudaErrorInitializationError:
    return "initialization error";
  case cudaErrorInvalidConfiguration:
    return "invalid device configuration argument";
  case cudaErrorInvalidDeviceFunction:
    return "invalid device function";
  case cudaErrorLaunchFailure:
    return "unspecified launch failure";
  default:
    return "unknown error";
  }
}

inline cudaError_t cudaMalloc(void **p, size_t sz) {
  if (!p)
    return detail::set_error(cudaErrorInvalidValue), cudaErrorInvalidValue;
  *p = std::malloc(sz);
  if (!p)
    return detail::set_error(cudaErrorMemoryAllocation),
           cudaErrorMemoryAllocation;
  return cudaSuccess;
}

inline cudaError_t cudaFree(void *p) {
  if (p)
    std::free(p);
  return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void *dst, const void *src, size_t n,
                              cudaMemCpyKind _mode) {
  if (n && (!dst || !src))
    return detail::set_error(cudaErrorInvalidValue), cudaErrorInvalidValue;
  std::memcpy(dst, src, n);
  return cudaSuccess;
}

inline cudaError_t cudaMemset(void *p, int v, size_t n) {
  if (n && !p)
    return detail::set_error(cudaErrorInvalidValue), cudaErrorInvalidValue;
  std::memset(p, v, n);
  return cudaSuccess;
}

inline cudaError_t cudaDeviceSynchronize() { return cudaPeekAtLastError(); }

//! Atomic device intrinsics
// atomicAdd/atomicSub/etc are __device__ free functions that operate on either
// global or shared memory. These operations need to be atomic because parallel
// threads may otherwise introduce race conditions. Here using std::atomic_ref
// which gives atomic access to non-owned memory.
template <class T> inline T atomicAdd(T *address, T val) {
  return std::atomic_ref<T>(*address).fetch_add(val);
}

template <class T> inline T atomicSub(T *address, T val) {
  return std::atomic_ref<T>(*address).fetch_sub(val);
}

template <class T> inline T atomicMax(T *address, T val) {
  std::atomic_ref<T> ref(*address);
  T old = ref.load();
  while (val > old && !ref.compare_exchange_weak(old, val)) {
  }
  return old;
}

template <class T> inline T atomicMin(T *address, T val) {
  std::atomic_ref<T> ref(*address);
  T old = ref.load();
  while (val < old && !ref.compare_exchange_weak(old, val)) {
  }
  return old;
}

template <class T> inline T atomicExch(T *address, T val) {
  return std::atomic_ref<T>(*address).exchange(val);
}

template <class T> inline T atomicCAS(T *address, T compare, T val) {
  std::atomic_ref<T>(*address).compare_exchange_strong(compare, val);
  return compare;
}

//! Math intrinsics:

// reciprocal sqrt:
inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }
inline double rsqrtf(double x) { return 1.0 / std::sqrt(x); }

// min/max:
#define DEFINE_MIN_FUNCTION(Type)                                              \
  inline Type min(Type a, Type b) { return a < b ? a : b; }

#define DEFINE_MAX_FUNCTION(Type)                                              \
  inline Type max(Type a, Type b) { return a > b ? a : b; }

DEFINE_MIN_FUNCTION(int)
DEFINE_MIN_FUNCTION(unsigned int)
DEFINE_MIN_FUNCTION(long long)
DEFINE_MIN_FUNCTION(float)
DEFINE_MIN_FUNCTION(double)

DEFINE_MAX_FUNCTION(int)
DEFINE_MAX_FUNCTION(unsigned int)
DEFINE_MAX_FUNCTION(long long)
DEFINE_MAX_FUNCTION(float)
DEFINE_MAX_FUNCTION(double)

// fast-math intrinsic aliases:
inline float __expf(float x) { return std::exp(x); }
inline float __logf(float x) { return std::log(x); }
inline float __log2f(float x) { return std::log2(x); }
inline float __sinf(float x) { return std::sin(x); }
inline float __cosf(float x) { return std::cos(x); }
inline float __powf(float a, float b) { return std::pow(a, b); }
inline float __fdividef(float a, float b) { return a / b; }
inline float __saturatef(float x) {
  return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
}

// cooperative fibers
//  each CUDA thread within a block is a coroutine, every fiber of a block
//  is multiplexed onto the single pool-worker OS thread that owns that block
//  __synchthreads is a cooperative yield
//
//  a coroutine is only ever created, resumed, and destroyed on a single OS
//  thread
inline constexpr size_t kFiberStackSize = 64 * 1024;
struct FiberBlock {
  unsigned int count = 0;         // fibers = threads per block
  std::vector<mco_coro *> fibers; // one coroutine per CUDA thread
  std::vector<dim3> thread_idx;   // each fiber's threadIdx
  std::function<void()> body;     // kernel body
};

// the default allocator is calloc, needlessly inefficient here
// kernel stacks don't need zeroing, so replace with malloc:
inline void *fiber_alloc(size_t size, void *) { return std::malloc(size); }
inline void fiber_dealloc(void *ptr, size_t, void *) { return std::free(ptr); }

// coroutine entry: run the block's kernel body once
// exception is caught inside the coroutine so never crosses a yield boundary
// device code exception becomes a launch failure
inline void fiber_entry(mco_coro *co) {
  FiberBlock *b = static_cast<FiberBlock *>(mco_get_user_data(co));
  try {
    b->body();
  } catch (...) {
    detail::set_error(cudaErrorLaunchFailure);
  }
}

// run every fiber of a block to completion using round-robin
// each pass resumes every living fiber once, so all fibers advance by one
// __syncthreads() phase per pass
inline void run_fiber_block(FiberBlock &b) {
  b.fibers.resize(b.count);
  for (unsigned int i = 0; i < b.count; i++) {
    mco_desc desc = mco_desc_init(fiber_entry, kFiberStackSize);
    desc.user_data = &b;
    desc.alloc_cb = fiber_alloc;
    desc.dealloc_cb = fiber_dealloc;
    mco_create(&b.fibers[i], &desc);
  }

  unsigned int remaining = b.count;
  while (remaining > 0) {
    for (unsigned int i = 0; i < b.count; i++) {
      mco_coro *co = b.fibers[i];
      if (mco_status(co) == MCO_DEAD) {
        continue;
      }
      threadIdx = b.thread_idx[i];
      mco_resume(co);
      if (mco_status(co) == MCO_DEAD) {
        remaining--;
      }
    }
  }

  for (unsigned int i = 0; i < b.count; i++) {
    mco_destroy(b.fibers[i]);
  }
}

//! Per-block context: shared block memory
struct BlockContext {
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
//
// The transpiler emits the natural C array type for array declarations
// e.g., `__shared__ int buf[8]` -> `get_shared_variable<int[8]>`
// so T here is an array type which decays to a pointer
template <class T> T &get_shared_variable(int id) {
  BlockContext *bc = current_thread_block;
  std::lock_guard<std::mutex> lock(bc->shared_mem_mutex);

  // Check if the variable has already been allocated by another thread
  auto it = bc->shared_mem.find(id);
  if (it != bc->shared_mem.end())
    return *static_cast<T *>(it->second);

  // allocate the variable
  T *p;
  if constexpr (std::is_trivial_v<T>) {
    // trivial types including C arrays get raw aligned storage
    // and is zero-initialised
    void *raw = ::operator new(sizeof(T), std::align_val_t(alignof(T)));
    std::memset(raw, 0, sizeof(T));
    p = static_cast<T *>(raw);
    // store deleter
    bc->deleters.push_back(
        [raw] { ::operator delete(raw, std::align_val_t(alignof(T))); });
  } else {
    // non array class keep normal construction/destruction
    p = new T();
    // capture the pointer so it can be correctly deleted during the
    // BlockContext destructor invocation
    bc->deleters.push_back([p] { delete p; });
  }
  bc->shared_mem.emplace(id, static_cast<void *>(p));
  return *p;
}

// emulates CUDA's `__syncthreads()`, using a cooperative yield from the running
// fiber back to the block's scheduler
inline void syncthreads() {
  mco_coro *co = mco_running();
  if (co)
    mco_yield(co);
}

//! Persistent worker pool
// spawning/joining OS threads every kernel launch is needless overhead
// so instead here we create a thread pool. Lazily creates its workers
// once and reuses them for every launch
class ThreadPool {
public:
  static ThreadPool &instance() {
    static ThreadPool pool;
    return pool;
  }

  // run task(i) for i in [0, count) across the pool until all blocks finish
  void parallel_for(unsigned int count,
                    const std::function<void(unsigned int)> &task) {
    if (count == 0) {
      return;
    }

    // publish the batch, then wake workers:
    {
      std::lock_guard<std::mutex> lock(mutex_);
      task_ = &task;
      cursor_.store(0, std::memory_order_relaxed);
      total_ = count;
      workers_done_.store(0, std::memory_order_relaxed);
      ++batch_;
    }
    cv_work_.notify_all();

    // wait until every worker has drained the batch, finished, and parked
    std::unique_lock<std::mutex> lock(mutex_);
    cv_done_.wait(lock, [this] {
      return workers_done_.load(std::memory_order_acquire) == worker_count_;
    });
    task_ = nullptr;
  }

  unsigned int size() const { return worker_count_; }

private:
  ThreadPool() {
    worker_count_ = std::thread::hardware_concurrency();
    if (worker_count_ == 0) {
      worker_count_ = 4; // fallback
    }
    workers_.reserve(worker_count_);
    for (unsigned int i = 0; i < worker_count_; i++) {
      workers_.emplace_back([this] { worker_loop(); });
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      ++batch_; // wake the workers so they can all see the stop
    }
    cv_work_.notify_all();
    for (auto &t : workers_) {
      t.join();
    }
  }

  void worker_loop() {
    unsigned long long seen = 0;
    while (true) {
      const std::function<void(unsigned int)> *task;
      unsigned int total;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_work_.wait(lock, [this, seen] { return stop_ || batch_ != seen; });
        if (stop_) {
          return;
        }
        seen = batch_;
        task = task_;
        total = total_;
      }

      // get a block and run ids until finished
      unsigned int i;
      while ((i = cursor_.fetch_add(1, std::memory_order_relaxed)) < total) {
        (*task)(i);
      }

      // update that this worker has drained the batch
      // last worker to finish wakes the caller
      if (workers_done_.fetch_add(1, std::memory_order_acq_rel) + 1 ==
          worker_count_) {
        std::lock_guard<std::mutex> lock(mutex_);
        cv_done_.notify_one();
      }
    }
  }

  std::vector<std::thread> workers_;
  unsigned int worker_count_ = 0;

  std::mutex mutex_;
  std::condition_variable cv_work_; // wake workers when a batch is posted
  std::condition_variable cv_done_; // wakes the caller when a batch completes

  const std::function<void(unsigned int)> *task_ = nullptr;
  unsigned int total_ = 0;
  std::atomic<unsigned int> cursor_{0}; // next block ID to claim
  std::atomic<unsigned int> workers_done_{
      0}; // workers that have drained the current batch
  unsigned long long batch_ = 0;
  bool stop_ = false;
};

//! parallel block executor:
// emulate GPU grid execution by running a kernel function, `f` across
// a specified number of thread blocks (`total_blocks`) dispatched onto the
// persistent ThreadPool
template <class KernelFunction>
void parallel_for_blocks(unsigned int total_blocks, KernelFunction f) {
  if (total_blocks == 0)
    return;
  ThreadPool::instance().parallel_for(total_blocks,
                                      std::function<void(unsigned int)>(f));
}

//! fast launch engine:
// run a grid of blocks concurrently across pool of workers
// doesn't support `__syncthreads()`  because there is no concurrent
// co-scheduling of threads within a block
template <class KernelFunction, class... KernelArgs>
void launch_fast(KernelFunction f, dim3 grid_dims, dim3 block_dims,
                 KernelArgs... args) {
  // pack args into tuple (for std::apply usage)
  auto args_tuple = std::make_tuple(args...);

  // flatten 3D grid and block structures into linear totals:
  unsigned int total_blocks = grid_dims.x * grid_dims.y * grid_dims.z;
  unsigned int threads_per_block = block_dims.x * block_dims.y * block_dims.z;

  if (total_blocks == 0 || threads_per_block == 0) {
    detail::set_error(cudaErrorInvalidConfiguration);
    return;
  }

  // distribute evenly:
  parallel_for_blocks(total_blocks, [&](unsigned int linear_block_id) {
    blockIdx = lin_to_dim(linear_block_id, grid_dims);
    blockDim = block_dims;
    gridDim = grid_dims;

    for (unsigned int id = 0; id < threads_per_block; ++id) {
      threadIdx = lin_to_dim(id, blockDim);
      try {
        std::apply(f, args_tuple);
      } catch (...) {
        detail::set_error(cudaErrorLaunchFailure);
        return;
      }
    }
  });
}

//! cooperative launch engine
// one OS thread per simulated CUDA thread within each block
// Threads within the same block share a common stack-allocated Barrier
// and BlockContext. This supports concurrent execution primitives like
// `__syncthreads()` and `__shared__` memory.
template <class KernelFunction, class... KernelArgs>
void launch_cooperative(KernelFunction f, dim3 grid_dims, dim3 block_dims,
                        KernelArgs... args) {
  auto args_tuple = std::make_tuple(args...);

  // flatten 3D grid and block structures into linear totals:
  unsigned int total_blocks = grid_dims.x * grid_dims.y * grid_dims.z;
  unsigned int threads_per_block = block_dims.x * block_dims.y * block_dims.z;

  if (total_blocks == 0 || threads_per_block == 0) {
    detail::set_error(cudaErrorInvalidConfiguration);
    return;
  }

  parallel_for_blocks(total_blocks, [&](unsigned int linear_block_id) {
    // per-block CUDA state
    // thread local on this worker and are shared by all of the block's fibers
    // only threadIdx varies per fiber
    BlockContext context;
    current_thread_block = &context;
    blockIdx = lin_to_dim(linear_block_id, grid_dims);
    blockDim = block_dims;
    gridDim = grid_dims;

    // build the fibers
    FiberBlock fb;
    fb.count = threads_per_block;
    fb.thread_idx.resize(threads_per_block);
    for (unsigned int i = 0; i < threads_per_block; i++) {
      fb.thread_idx[i] = lin_to_dim(i, block_dims);
    }
    fb.body = [&] { std::apply(f, args_tuple); };
    run_fiber_block(fb);
  });
}

} // namespace cucpu