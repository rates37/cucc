// Benchmark for the block-executor / worker-pool
//
// metric important here is per-launch overhead (how much fixed cost
// each kernel launch pays independent of the actual work). With a spawn-per-launch
// executor that cost is dominated by creating and joining OS threads on every
// launch, but a persistent pool amortizes it away. So the headline scenarios launch
// a tiny kernel many times. 
//
// Output is one line per scenario:  "<tag> <ms/launch>"  (plus a summary line).
#include <chrono>
#include <cstdio>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) {
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// tiny fast-path kernel (routed to launch_fast)
__global__ void inc_fast(int *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    a[i] += 1;
}

// tiny cooperative-path kernel (__shared__ + __syncthreads -> launch_cooperative)
__global__ void inc_coop(int *a, int n) {
  __shared__ int s[64];
  int t = threadIdx.x;
  int i = blockIdx.x * blockDim.x + t;
  s[t] = (i < n) ? a[i] : 0;
  __syncthreads();
  if (i < n)
    a[i] = s[t] + 1;
}

// median of repeated timings keeps scheduling noise from dominating
static double median(double *v, int m) {
  for (int i = 1; i < m; i++) // tiny insertion sort, m is small
    for (int j = i; j > 0 && v[j - 1] > v[j]; j--) {
      double tmp = v[j - 1];
      v[j - 1] = v[j];
      v[j] = tmp;
    }
  return (m & 1) ? v[m / 2] : 0.5 * (v[m / 2 - 1] + v[m / 2]);
}

template <class Launch>
static double per_launch_ms(Launch launch, int iters, int reps) {
  double samples[16];
  launch(); // warmup
  cudaDeviceSynchronize();
  for (int r = 0; r < reps; r++) {
    auto t0 = clk::now();
    for (int k = 0; k < iters; k++) {
      launch();
      cudaDeviceSynchronize();
    }
    samples[r] = ms_since(t0) / iters;
  }
  return median(samples, reps);
}

int main() {
  const int reps = 5;

  // Scenario A: many small FAST launches (grid=64, block=64)
  double a_ms;
  {
    const int n = 64 * 64, threads = 64, blocks = (n + threads - 1) / threads;
    int *d;
    cudaMalloc((void **)&d, n * sizeof(int));
    cudaMemset(d, 0, n * sizeof(int));
    a_ms = per_launch_ms([&] { inc_fast<<<blocks, threads>>>(d, n); }, 2000,
                         reps);
    cudaFree(d);
    std::printf("A many_small_fast %.5f\n", a_ms);
  }

  // Scenario B: many small COOPERATIVE launches. Small blocks (grid=16, block=8)
  double b_ms;
  {
    const int n = 16 * 8, threads = 8, blocks = (n + threads - 1) / threads;
    int *d;
    cudaMalloc((void **)&d, n * sizeof(int));
    cudaMemset(d, 0, n * sizeof(int));
    b_ms = per_launch_ms([&] { inc_coop<<<blocks, threads>>>(d, n); }, 250,
                         reps);
    cudaFree(d);
    std::printf("B many_small_coop %.5f\n", b_ms);
  }

  // Scenario C: large single FAST launch (throughput / regression guard). Work
  // dominates launch overhead -> should be flat across the change.
  double c_ms;
  {
    const int n = 1 << 22, threads = 256, blocks = (n + threads - 1) / threads;
    int *d;
    cudaMalloc((void **)&d, n * sizeof(int));
    cudaMemset(d, 0, n * sizeof(int));
    c_ms = per_launch_ms([&] { inc_fast<<<blocks, threads>>>(d, n); }, 40, reps);
    cudaFree(d);
    std::printf("C big_fast %.5f\n", c_ms);
  }

  std::printf("summary A=%.5f B=%.5f C=%.5f (ms/launch)\n", a_ms, b_ms, c_ms);
  return 0;
}
