// Inclusive prefix sum (scan) over a single block using the Hillis-Steele
// algorithm. Uses double-buffered __shared__ memory + __syncthreads(), so it
// runs on the cooperative launch engine. Keeping it to one block makes the
// result fully deterministic without a cross-block fixup pass.
//
// Input is all ones, so the inclusive scan of element i is (i + 1).
// Expected output: "1 128 256"
//   out[0] = 1, out[127] = 128, out[255] = 256
#include <cstdio>

#define N 256

__global__ void scan_inclusive(const int *in, int *out, int n) {
  __shared__ int a[N];
  __shared__ int b[N];
  int t = threadIdx.x;

  a[t] = in[t];
  __syncthreads();

  // Ping-pong between the two buffers, doubling the offset each step.
  // Note: need to address the arrays via &x[0] rather than relying on array-to-pointer
  // decay -- __shared__ arrays are lowered to std::array, which does not decay.
  // this needs to be fixed (ideally)
  int *src = &a[0];
  int *dst = &b[0];
  for (int offset = 1; offset < n; offset <<= 1) {
    if (t >= offset)
      dst[t] = src[t] + src[t - offset];
    else
      dst[t] = src[t];
    __syncthreads();

    int *tmp = src;
    src = dst;
    dst = tmp;
    __syncthreads();
  }

  out[t] = src[t];
}

int main() {
  const int n = N;
  int h[N];
  for (int i = 0; i < n; i++)
    h[i] = 1;

  int *din, *dout;
  cudaMalloc((void **)&din, n * sizeof(int));
  cudaMalloc((void **)&dout, n * sizeof(int));
  cudaMemcpy(din, h, n * sizeof(int), cudaMemcpyHostToDevice);

  scan_inclusive<<<1, n>>>(din, dout, n);
  cudaDeviceSynchronize();

  int r[N];
  cudaMemcpy(r, dout, n * sizeof(int), cudaMemcpyDeviceToHost);
  std::printf("%d %d %d\n", r[0], r[127], r[255]);

  cudaFree(din);
  cudaFree(dout);
  return 0;
}