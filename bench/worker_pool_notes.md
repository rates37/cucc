# Persistent Worker Pool

Commit 331eac5 replaced the spawn a fresh set of `std::thread`s per launch executor in `parallel_for_blocks` with a persistent `ThreadPool`, created once and reused on subsequent launches.

### Method:

Metric used: ms per launch, kernel + `cudaDeviceSynchronize()`, median of 5 reps after a warmup launch.

Scenarios:

- Scenario A: `many_small_fast`: tiny, fast kernel, grid 64 x block 64, 2000 launches. Pure per-thread overhead

- Scenario B: `many_small_coop`: tiny cooperative kernel (**shared** + \_\_syncthreads), grid 16 x block 8. Small blocks on purpose so that the per-CUDA-thread inner spawn doesn't hide the outer pool overhead which will be changed in a later change

- Scenario C: `big_fast`: one large fast launch, n = $2^22$, grid 16384 x block 256

### Results (ms per launch):

| Scenario            | Before (spawn/launch) | After(pool) | Change |
| ------------------- | --------------------- | ----------- | ------ |
| A `many_small_fast` | 0.10940               | 0.04601     | -58%   |
| B `many_small_coop` | 1.65518               | 1.65544     | 0%     |
| C `big_fast`        | 4.02314               | 5.09441     | +27%   |

### Analysis:

Every launch previously called `std::thread` and joined them. Each creation is a `clone()` syscall plus kernel-side task/stack setup, scheduler enqueue, etc. Each `join` is a futex wait plus teardown. For each thread this may be on the order of 10s of microseconds, when accounting for number of threads, and launches, this adds to a significant overhead. Worst of all it scales linearly with the number of launches, bad for long-running programs. Scenario A is a good test of this overhead and shows a 58% reduction in launch time after the change.

After this change, workers are created once and park on a condition variable. A launch now: publishes the batch and does one `notify_all` (futex wake), lets workers pull block IDs off a shared atomic cursor (counter), and waits for them all to finish. This converts the per-paunch cost from thread creation to thread **wakeup**, hence the nearly 2.5x speedup.

Switching the completion counter from a mutex-guarded lock to a lock-free atomic (only the last worker takes the lock, to signal) did not improve performance noticeably.

Scenario B didn't change at all because the cooperative path still spawns one OS thread per CUDA thread per block, and joins them. The pool only removed the outer worker spawn, the inner thread explosion risk is still there and is unchanged. 


### Notes on mistakes made in implementation:

An early version of this threadpool implementation had completion signalled when the last block finished, which resulted in a segfault. A worker lagging the batch could call `cursor_.fetch_add()` after the next launch had reset the cursor and reused the pool state, re-entering its loop with a stale (freed) `task` pointer. The fix was to signal completion only once all the workers had drained the batch AND parked, so no worker can still be touching the shared state when the next launch reuses it. 
