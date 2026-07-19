// Tiled matrix transpose using 2D __shared__ memory. Exercises multidimensional
// __shared__ arrays (rewritten to native C arrays), 2D indexing, and
// __syncthreads() so this runs on the cooperative launch engine.
//
// The kernel result is checked against a serial CPU reference inside main().
// Expected output: "transpose: OK"
#include <cstdio>

#define TILE 16

// Transpose a WIDTH x HEIGHT matrix (row-major) into a HEIGHT x WIDTH matrix.
__global__ void transpose(const float *in, float *out, int width, int height) {
  __shared__ float tile[TILE][TILE];

  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  if (x < width && y < height)
    tile[threadIdx.y][threadIdx.x] = in[y * width + x];
  __syncthreads();

  // Write the tile back transposed: the block's (x,y) maps to (y,x).
  int tx = blockIdx.y * TILE + threadIdx.x;
  int ty = blockIdx.x * TILE + threadIdx.y;
  if (tx < height && ty < width)
    out[ty * height + tx] = tile[threadIdx.x][threadIdx.y];
}

int main() {
  const int width = 64, height = 32, n = width * height;
  float *hin = new float[n];
  for (int i = 0; i < n; i++)
    hin[i] = (float)i;

  float *din, *dout;
  cudaMalloc((void **)&din, n * sizeof(float));
  cudaMalloc((void **)&dout, n * sizeof(float));
  cudaMemcpy(din, hin, n * sizeof(float), cudaMemcpyHostToDevice);

  dim3 block(TILE, TILE);
  dim3 grid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);
  transpose<<<grid, block>>>(din, dout, width, height);
  cudaDeviceSynchronize();

  float *hout = new float[n];
  cudaMemcpy(hout, dout, n * sizeof(float), cudaMemcpyDeviceToHost);

  int mismatches = 0;
  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++)
      if (hout[x * height + y] != hin[y * width + x])
        mismatches++;
  std::printf("transpose: %s\n", mismatches == 0 ? "OK" : "FAIL");

  cudaFree(din);
  cudaFree(dout);
  delete[] hin;
  delete[] hout;
  return 0;
}