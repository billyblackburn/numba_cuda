# Example 1.4: Add arrays with grid striding
import numpy as np
import numba
from numba import cuda


@cuda.jit
def add_array_gs(a, b, c):
    i_start = cuda.grid(1)
    threads_per_grid = cuda.blockDim.x * cuda.gridDim.x
    for i in range(i_start, a.size, threads_per_grid):
        c[i] = a[i] + b[i] 
# This way, if the total number of threads in the grid 
# (threads_per_grid = blockDim.x * gridDim.x) is smaller than the number 
# of elements of the array, as soon as the kernel is done processing the 
# index cuda.grid(1) it will process the index cuda.grid(1) + threads_per_grid
#  and so on until all array elements have been processed.

N = 1_000_000
a = np.arange(N, dtype=np.float32)
b = np.arange(N, dtype=np.float32)

dev_a = cuda.to_device(a)
dev_b = cuda.to_device(b)


dev_c = cuda.device_array_like(a)

threads_per_block = 256
blocks_per_grid_gs = 32 * 80  # Use 32 * multiple of streaming multiprocessors
# 32 * 80 * 256 < 1_000_000 so one thread will process more than one array element

add_array_gs[blocks_per_grid_gs, threads_per_block](dev_a, dev_b, dev_c)
c = dev_c.copy_to_host()
ac = np.allclose(a + b, c)
print(c,ac)