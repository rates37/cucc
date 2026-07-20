// Error-surface fixture. Confirms the CUDA-aligned error API:
//   * clean launch leaves no error
//   * invalid launch configuration (zero block dim) reports cudaErrorInvalidConfiguration
//   * exception thrown in device code is captured as cudaErrorLaunchFailure
//     instead of crashing the process -- on both the fast path and the
//     cooperative path (where a naive impl would deadlock at __syncthreads())
//   * cudaGetLastError clears the sticky error; cudaDeviceSynchronize / peek do not
//
// Expected output:
//   clean=cudaSuccess
//   config=cudaErrorInvalidConfiguration
//   fastthrow=cudaErrorLaunchFailure
//   string=unspecified launch failure
//   sticky=cudaErrorLaunchFailure,cudaSuccess
//   coopthrow=cudaErrorLaunchFailure
#include <cstdio>
#include <stdexcept>

__global__ void ok_kernel(int *out) { out[threadIdx.x] = (int)threadIdx.x; }

// fast-path kernel (no __shared__ / __syncthreads) that throws in one thread
__global__ void fast_thrower(int *out) {
  int t = threadIdx.x;
  if (t == 3)
    throw std::runtime_error("device fault");
  out[t] = t;
}

// cooperative-path kernel: throws before the barrier, which must not deadlock
// the block's other threads.
__global__ void coop_thrower(int *out) {
  __shared__ int buf[8];
  int t = threadIdx.x;
  if (t == 3)
    throw std::runtime_error("device fault");
  buf[t] = t;
  __syncthreads();
  out[t] = buf[t];
}

int main() {
  int *d;
  cudaMalloc((void **)&d, 8 * sizeof(int));

  // clean launch -> no error
  ok_kernel<<<1, 8>>>(d);
  cudaError_t e1 = cudaDeviceSynchronize();
  std::printf("clean=%s\n", cudaGetErrorName(e1));

  // invalid configuration: zero-sized block, kernel never runs
  ok_kernel<<<1, 0>>>(d);
  cudaError_t e2 = cudaGetLastError();
  std::printf("config=%s\n", cudaGetErrorName(e2));

  // exception on the fast path is captured, process survives
  fast_thrower<<<1, 8>>>(d);
  cudaError_t e3 = cudaDeviceSynchronize();
  std::printf("fastthrow=%s\n", cudaGetErrorName(e3));
  std::printf("string=%s\n", cudaGetErrorString(e3));

  // the error is sticky until cudaGetLastError clears it
  cudaError_t e4 = cudaGetLastError(); // returns it and clears
  cudaError_t e5 = cudaGetLastError(); // now cleared
  std::printf("sticky=%s,%s\n", cudaGetErrorName(e4), cudaGetErrorName(e5));

  // exception on the cooperative path: must capture, not deadlock
  coop_thrower<<<1, 8>>>(d);
  cudaError_t e6 = cudaDeviceSynchronize();
  std::printf("coopthrow=%s\n", cudaGetErrorName(e6));

  cudaFree(d);
  return 0;
}