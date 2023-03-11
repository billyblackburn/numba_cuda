import numpy as np
import numba
from numba import cuda

# Example 1.2: Add arrays
@cuda.jit
def add_array(a, b, c):
    i = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
# but now we have an issue where some threads would overflow 
# the array,# since the array has 20 elements and i goes up to
#  32-1. 
# The solution is simple: for those threads, don't do anything!
    if i < a.size:
        c[i] = a[i] + b[i]

N = 1_000_000
a = np.arange(N, dtype=np.float32)
b = np.arange(N, dtype=np.float32)

dev_a = cuda.to_device(a)
dev_b = cuda.to_device(b)


dev_c = cuda.device_array_like(a)

threads_per_block = 256
blocks_per_grid = (N + (threads_per_block - 1)) // threads_per_block
# Note that blocks_per_grid == ceil(N / threads_per_block)
# ensures that blocks_per_grid * threads_per_block >= N

print(a)


add_array[blocks_per_grid, threads_per_block](dev_a, dev_b, dev_c)

c = dev_c.copy_to_host()
np.allclose(a + b, c)

#  True
print(c)

#  [ 0.  2.  4.  6.  8. 10. 12. 14. 16. 18. 20. 
#    22. 24. 26. 28. 30. 32. 34. 36. 38.]