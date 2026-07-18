#include <cmath>
#include <cstdio>
#include <vector>

// usage: bin/cucc examples/vecadd.cu

__global__ void vecadd(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] + b[i];
}

int main() {
  // create input data:
  const int n = 1 << 20;
  std::vector<float> a(n), b(n), c(n);
  for (int i = 0; i < n; i++) {
    a[i] = i * 0.5f;
    b[i] = i;
  }

  // allocate space on 'GPU':
  float *da, *db, *dc;
  cudaMalloc((void **)&da, n * sizeof(float));
  cudaMalloc((void **)&db, n * sizeof(float));
  cudaMalloc((void **)&dc, n * sizeof(float));
  cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  // run kernels:
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  vecadd<<<blocks, threads>>>(da, db, dc, n);
  cudaDeviceSynchronize();

  // copy data back to 'cpu':
  cudaMemcpy(c.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost);

  // check results:
  double maxerr = 0;
  for (int i = 0; i < n; i++) {
    maxerr = std::fmax(maxerr, std::fabs(c[i] - (a[i] + b[i])));
  }
  std::printf("vecadd n=%d, max|err| = %g\n", n, maxerr);

  // free GPU resources:
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);

  return 0;
}
