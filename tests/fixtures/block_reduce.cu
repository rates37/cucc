// Cooperative-path integration fixture: a per-block sum reduction using
// __shared__ memory and __syncthreads(). Exercises launch_cooperative.
//
// Expected output: "28 92"
//   block 0 sums indices 0..7  = 28
//   block 1 sums indices 8..15 = 92
#include <cstdio>

__global__ void block_reduce(const int *in, int *out) {
  __shared__ int buf[8];
  int t = threadIdx.x;
  buf[t] = in[blockIdx.x * blockDim.x + t];
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s)
      buf[t] += buf[t + s];
    __syncthreads();
  }

  if (t == 0)
    out[blockIdx.x] = buf[0];
}

int main() {
  const int blocks = 2, threads = 8, n = blocks * threads;
  int h[16];
  for (int i = 0; i < n; i++)
    h[i] = i;

  int *din, *dout;
  cudaMalloc((void **)&din, n * sizeof(int));
  cudaMalloc((void **)&dout, blocks * sizeof(int));
  cudaMemcpy(din, h, n * sizeof(int), cudaMemcpyHostToDevice);

  block_reduce<<<blocks, threads>>>(din, dout);
  cudaDeviceSynchronize();

  int r[2];
  cudaMemcpy(r, dout, blocks * sizeof(int), cudaMemcpyDeviceToHost);
  std::printf("%d %d\n", r[0], r[1]);

  cudaFree(din);
  cudaFree(dout);
  return 0;
}
