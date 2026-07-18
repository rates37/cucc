// Histogram: every thread bumps the bin for its element with atomicAdd. This is
// a fast-path kernel (no __shared__ / __syncthreads()), so blocks run
// concurrently across worker threads and atomicAdd contention is a possibility
//
// Input value at index i is (i % NBINS), so each of the NBINS bins receives
// exactly N / NBINS hits.
// Expected output: "100 100 100 100 100 100 100 100 100 100"
#include <cstdio>

#define NBINS 10

__global__ void histogram(const int *in, int *bins, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    atomicAdd(&bins[in[i]], 1);
}

int main() {
  const int n = 1000;
  int h[1000];
  for (int i = 0; i < n; i++)
    h[i] = i % NBINS;

  int *din, *dbins;
  cudaMalloc((void **)&din, n * sizeof(int));
  cudaMalloc((void **)&dbins, NBINS * sizeof(int));
  cudaMemcpy(din, h, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(dbins, 0, NBINS * sizeof(int));

  const int threads = 128;
  const int blocks = (n + threads - 1) / threads;
  histogram<<<blocks, threads>>>(din, dbins, n);
  cudaDeviceSynchronize();

  int r[NBINS];
  cudaMemcpy(r, dbins, NBINS * sizeof(int), cudaMemcpyDeviceToHost);
  for (int b = 0; b < NBINS; b++)
    std::printf("%d%s", r[b], b + 1 < NBINS ? " " : "\n");

  cudaFree(din);
  cudaFree(dbins);
  return 0;
}