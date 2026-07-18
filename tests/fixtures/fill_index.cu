// Fast-path integration fixture: each thread writes its global index * 10.
// No __shared__ / __syncthreads(), so this exercises launch_fast end-to-end.
// Deterministic output: one integer per line, value == index * 10.
#include <cstdio>

__global__ void fill(int *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = i * 10;
}

int main() {
  const int n = 64;
  int *d;
  cudaMalloc((void **)&d, n * sizeof(int));

  const int threads = 8;
  const int blocks = (n + threads - 1) / threads;
  fill<<<blocks, threads>>>(d, n);
  cudaDeviceSynchronize();

  int h[64];
  cudaMemcpy(h, d, n * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++)
    std::printf("%d\n", h[i]);

  cudaFree(d);
  return 0;
}
