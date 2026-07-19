"""
End-to-end tests for the full cucc toolchain (transpile + compile + run)
"""

import os
import shutil
import subprocess
import tempfile
import unittest

from tests._util import CUCC, FIXTURES_DIR, heavy_enabled


def compile_and_run(source_path, run_args=None, timeout=60):
    """Compile `source_path` with bin/cucc and run the resulting binary

    Returns the captured stdout (str). Raises AssertionError with the compiler
    or program output on failure so failed test messages are actionable.
    """
    workdir = tempfile.mkdtemp(prefix="cucc_test_")
    try:
        binary = os.path.join(workdir, "prog")
        build = subprocess.run(
            [CUCC, "-o", binary, source_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if build.returncode != 0:
            raise AssertionError(
                f"cucc failed to build {source_path} (exit: {build.returncode})\n\n"
                f"stdout:\n{build.stdout}\n\n"
                f"stderr:\n{build.stderr}"
            )
        run = subprocess.run(
            [binary, *(run_args or [])], capture_output=True, text=True, timeout=timeout
        )
        if run.returncode != 0:
            raise AssertionError(
                f"program {binary} exited with {run.returncode}\n\n"
                f"stdout:\n{run.stdout}\n\n"
                f"stderr:\n{run.stderr}"
            )
        return run.stdout
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


class DriverContractTests(unittest.TestCase):
    def test_cucc_reports_missing_input(self):
        # not providing a .cu file as input should be an error:
        r = subprocess.run([CUCC], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)


class FastPathTests(unittest.TestCase):
    # kernels routed to launch_fast
    def test_fill_index(self):
        out = compile_and_run(os.path.join(FIXTURES_DIR, "fill_index.cu"))
        values = [int(l) for l in out.split() if l.strip()]
        self.assertEqual(values, [i * 10 for i in range(64)])


class CooperativePathTests(unittest.TestCase):
    # kernels routed to launch_cooperative
    def test_block_reduce(self):
        out = compile_and_run(os.path.join(FIXTURES_DIR, "block_reduce.cu"))
        self.assertEqual(out.split(), ["28", "92"])


class GeneralKernelTests(unittest.TestCase):
    def test_histogram(self):
        out = compile_and_run(os.path.join(FIXTURES_DIR, "histogram.cu"))
        self.assertEqual(out.split(), ["100"] * 10)

    def test_reduction(self):
        out = compile_and_run(os.path.join(FIXTURES_DIR, "reduction.cu"))
        self.assertEqual(out.split(), ["523776"])

    def test_scan(self):
        out = compile_and_run(os.path.join(FIXTURES_DIR, "scan.cu"))
        self.assertEqual(out.split(), ["1", "128", "256"])

    def test_stencil(self):
        out = compile_and_run(os.path.join(FIXTURES_DIR, "stencil.cu"))
        self.assertEqual(out.split(), ["stencil:", "OK"])

    def test_transpose(self):
        out = compile_and_run(os.path.join(FIXTURES_DIR, "transpose.cu"))
        self.assertEqual(out.split(), ["transpose:", "OK"])

    def test_shared_decay(self):
        out = compile_and_run(os.path.join(FIXTURES_DIR, "shared_decay.cu"))
        self.assertEqual(
            out.split(),
            [
                "7",
                "6",
                "5",
                "4",
                "3",
                "2",
                "1",
                "0",
                "15",
                "14",
                "13",
                "12",
                "11",
                "10",
                "9",
                "8",
            ],
        )
    
    def test_device_math(self):
        out = compile_and_run(os.path.join(FIXTURES_DIR, "device_math.cu"))
        self.assertEqual(
            out.split(),
            ["5", "7", "9", "11", "13", "15", "17", "19"]
        )


@unittest.skipUnless(heavy_enabled(), "heavy test, set CUCC_RUN_HEAVY=1 to enable")
class StressTests(unittest.TestCase):
    # only run if
    def test_large_vecadd(self):
        # Larger working set than the default example
        src = """
#include <cmath>
#include <cstdio>
#include <vector>
__global__ void vecadd(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
int main() {
  const int n = 1 << 24;
  std::vector<float> a(n), b(n), c(n);
  for (int i = 0; i < n; i++) { a[i] = i * 0.5f; b[i] = i; }
  float *da, *db, *dc;
  cudaMalloc((void **)&da, n * sizeof(float));
  cudaMalloc((void **)&db, n * sizeof(float));
  cudaMalloc((void **)&dc, n * sizeof(float));
  cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  int threads = 256, blocks = (n + threads - 1) / threads;
  vecadd<<<blocks, threads>>>(da, db, dc, n);
  cudaDeviceSynchronize();
  cudaMemcpy(c.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost);
  double maxerr = 0;
  for (int i = 0; i < n; i++)
    maxerr = std::fmax(maxerr, std::fabs(c[i] - (a[i] + b[i])));
  std::printf("ok maxerr=%g\\n", maxerr);
  cudaFree(da); cudaFree(db); cudaFree(dc);
  return 0;
}
"""
        workdir = tempfile.mkdtemp(prefix="cucc_stress_")
        try:
            cu = os.path.join(workdir, "big.cu")
            with open(cu, "w") as f:
                f.write(src)
            out = compile_and_run(cu, timeout=5)
            self.assertIn("maxerr=0", out)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
