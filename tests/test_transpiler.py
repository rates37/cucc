"""
Unit tests for the cucc transpiler in isolation
"""

import unittest
from tests._util import load_transpiler

T = load_transpiler()


class ScanForKernelsTests(unittest.TestCase):
    def test_simple_kernel_is_not_cooperative(self):
        src = "__global__ void vecadd(const float *a, float *c, int n) { c[0] = a[0]; }"
        self.assertEqual(T.scan_for_kernels(src), {"vecadd": False})

    def test_syncthreads_requires_cooperative(self):
        src = "__global__ void k(int *o) { __syncthreads(); o[0] = 1; }"
        self.assertEqual(T.scan_for_kernels(src), {"k": True})

    def test_shared_requires_cooperative(self):
        src = "__global__ void k(int *o) { __shared__ int buf[4]; o[0] = buf[0]; }"
        self.assertEqual(T.scan_for_kernels(src), {"k": True})

    def test_multiple_kernels(self):
        src = (
            "__global__ void a(int *o) { o[0] = 0; }\n"
            "__global__ void b(int *o) { __syncthreads(); }\n"
        )
        self.assertEqual(T.scan_for_kernels(src), {"a": False, "b": True})


class RewriteSharedMemoryTests(unittest.TestCase):
    def test_1d_array(self):
        out = T.rewrite_shared_memory("__shared__ int buf[8];")
        self.assertEqual(
            out,
            "auto& buf = ::cucpu::get_shared_variable<int[8]>(1);",
        )

    def test_multidim_array_preserves_dim_order(self):
        out = T.rewrite_shared_memory("__shared__ float matrix[16][32];")
        self.assertEqual(
            out,
            "auto& matrix = ::cucpu::get_shared_variable<float[16][32]>(1);",
        )
    
    def test_scalar_shared_has_no_array_suffix(self):
        out = T.rewrite_shared_memory("__shared__ int total;")
        self.assertEqual(
            out,
            "auto& total = ::cucpu::get_shared_variable<int>(1);"
        )

    def test_unique_ids_are_assigned(self):
        out = T.rewrite_shared_memory("__shared__ int a[2]; __shared__ int b[2];")
        self.assertIn("<int[2]>(1);", out)
        self.assertIn("<int[2]>(2);", out)


class RewriteKernelLaunchesTests(unittest.TestCase):
    def test_fast_launch(self):
        src = "vecadd<<<blocks, threads>>>(da, db, dc, n);"
        out = T.rewrite_kernel_launches(src, {"vecadd": False})
        self.assertEqual(
            out,
            "::cucpu::launch_fast(vecadd, blocks, threads, da, db, dc, n);",
        )

    def test_cooperative_launch(self):
        src = "k<<<2, 4>>>(d);"
        out = T.rewrite_kernel_launches(src, {"k": True})
        self.assertEqual(out, "::cucpu::launch_cooperative(k, 2, 4, d);")

    def test_unknown_kernel_defaults_to_cooperative(self):
        # A launched kernel with no local __global__ definition falls back to the
        # cooperative engine
        src = "mystery<<<1, 1>>>(x);"
        out = T.rewrite_kernel_launches(src, {})
        self.assertEqual(out, "::cucpu::launch_cooperative(mystery, 1, 1, x);")

    def test_no_arguments(self):
        src = "k<<<1, 1>>>();"
        out = T.rewrite_kernel_launches(src, {"k": False})
        self.assertEqual(out, "::cucpu::launch_fast(k, 1, 1);")

    def test_left_shift_is_not_a_launch(self):
        # `<<` used as a bit-shift (not a `<<<` launch) must be left untouched
        src = "int y = a << 3;"
        out = T.rewrite_kernel_launches(src, {})
        self.assertEqual(out, src)


class TranspileSourceTests(unittest.TestCase):
    def test_prelude_is_prepended(self):
        out = T.transpile_source("int main() { return 0; }")
        self.assertTrue(out.startswith(T.CUCC_PRELUDE))
        self.assertIn('#include "cucpu.hpp"', out)

    def test_end_to_end_fast_kernel(self):
        src = (
            "__global__ void fill(int *o, int n) {\n"
            "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "  if (i < n) o[i] = i;\n"
            "}\n"
            "int main() { fill<<<2, 4>>>(d, 8); }\n"
        )
        out = T.transpile_source(src)
        self.assertIn("::cucpu::launch_fast(fill, 2, 4, d, 8);", out)
        self.assertIn("void fill(int *o, int n)", out)
