# Fibers for Cooperative Launches

Commit `28033e25e9cc95ba056a69eeec54f9d4b85b1fd6` replaces the "one OS thread per CUDA thread" in the cooperative launch path with fibers / stackful coroutines. Each block runs on a single pool-worker (OS thread), and its CUDA threads become cooperative yield back to a per-block scheduler.

Context switch implemented using `minicoro.h`. The first implementation attempt used ucontext.h, but it was deprecated/removed in newer macOS versions, so defeated the ideology of this project (to be able to 'run'/emulate CUDA on any platform).

### Method:

Metric used: ms per launch, kernel + `cudaDeviceSynchronize()`, median of reps after warmup

Scenarios:

A-C are same as worker_pool_notes.md, scenario D is new.

- Scenario A: `many_small_fast`: tiny, fast kernel, grid 64 x block 64, 2000 launches. Pure per-thread overhead

- Scenario B: `many_small_coop`: tiny cooperative kernel (**shared** + \_\_syncthreads), grid 16 x block 8. Small blocks on purpose so that the per-CUDA-thread inner spawn doesn't hide the outer pool overhead which will be changed in a later change

- Scenario C: `big_fast`: one large fast launch, n = $2^22$, grid 16384 x block 256

- Scenario D: `big_block_coop`: grid 8 x block 256 are cooperative, so would cause a thread explosion in original implementation (over 1000 OS threads)

### Results (ms per launch):

| Scenario            | Before (spawn/launch) | After(pool) | Change |
| ------------------- | --------------------- | ----------- | ------ |
| A `many_small_fast` | 0.04566               | 0.04509     | -1%    |
| B `many_small_coop` | 1.69841               | 0.03893     | -98%   |
| C `big_fast`        | 5.29991               | 5.14334     | -3%    |
| D `big_block_coop`  | 33.62861              | 3.91655     | -88%   |

### Analysis:

The old cooperative launch did `std::thread` * `threads_per_block` per block, per launch, then joined them. Each is a `clone()` syscall with a kernel task struct + 8MB virtual stack, scheduler enqueue, etc. Each `join` is a futex wait plus teardown (like mentioned in the worker pool notes). This scales with the number of CUDA threads. This means ordinary CUDA thread sizes (such as a 256 thread block) meant thousands of OS threads across the grid, either thrashing or hitting the OS thread limit. This really limited the practicality of `cucc` for emulating real CUDA workloads. Scenario D is a test for this overhead and shows a 88% reduction in launch time after the change.

A fiber is a stack plus a saved register set. Creating one is just a memory allocation + context initialisation with no kernel involvement. This puts the responsibility of context switching in user space, which naturally makes the code a little less pretty, but the upside is that it is both much faster and handles larger workloads. All fibers in a block are run on a single OS thread that owns the entire block. This got a ~98% speedup for scenario B and ~88% for scenario D. Scenario A and C didn't change (as expected) because they aren't cooperative launches, so the fiber implementation doesn't affect these paths.

The `run_fiber_block` function uses round-robin passes to schedule the fibers. Each pass runs every unfinished fiber once, and a fiber runs until it either finishes, or yields back to the scheduler via `__syncthreads()`. 

Because a pass touches every fiber before the next pass starts, all fibers advance by exactly one sync phase per pass, which IS the `__syncthreads()` barrier. And because of this, there is no longer a need for a lock, condition variable, or other routine synchronisation mechanism. The `__syncthreads()` function is now implemented as a `mco_yield(mco_running());` call, which saves the current fiber's context and returns to the scheduler. The scheduler then runs the next fiber in the round-robin order. A `null` from `mco_running()` makes `__syncthreads()` a no-op, whereas the old implementation needed a full condition-variable, `Barrier` to get the same effect across threads.


#### Switch from `ucontext.h` to `minicoro.h`

The first attempt at this change used `ucontext.h` to implement the fibers, but this was deprecated and removed in POSIX.1-2008, and entirely unavailable on some platforms. The final implementation uses `minicoro.h`, which is a small, header-only library that implements stackful coroutines using hand-written assembly to context switch. 

Initially after switching to minicoro, the performance of the cooperative launch path was worse than the `ucontext.h` implementation by around a factor of 10 (it essentially showed no speedup from the OS-thread based approach). The root cause turned out to be that the minicoro's default allocator is `calloc`, so every fiber stack is zero-filled when created. A 256-thread block allocates 256*64kB = 16MB of zeroed memory, every launch. This was a waste, as a fresh stack doesn't need zeroing.

Switching to `malloc`/`free` via `mco_desc.alloc_cb` and `mco_desc.dealloc_cb` fixed this. This is a safe choice to make since the `mco_init` function `memset`s the control struct regardless of allocator anyway, and neither the stack nor the storage buffer needed pre-zeroing. After this change, the performance of the minicoro implementation was about as fast as the `ucontext.h` one, with better portability.

The minicoro + malloc implementation is actually faster in scenario D when compared to the `ucontext.h` one. This is because glibc's `swapcontext` performs a `sigprocmask` syscall on every context switch to save/restore the signal mask, whereas minicoro's assembly implementation is a purely user-space save/restore operation. Scenario D does ~512 switches per block per launch, so eliminating a syscall per switch is a noticeable speedup.