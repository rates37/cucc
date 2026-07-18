# cucc
cucc - CUDA Universal CPU Compiler

Write kernels that *read* like CUDA, compile them with `cucc`, and run a a normal multi-threaded binary. No GPU required.


## Requirements:

* A C++ compiler with C++17 support
* Python3


## Quickstart:

Use `bin/cucc` just like you would `nvcc`:

```bash
$ bin/cucc -o vecadd examples/vecadd.cu
```
