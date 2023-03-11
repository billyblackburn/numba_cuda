import numpy as np
import numba
from numba import cuda

# Example 1.1: Add scalars

# Our first lesson is that kernels 
# (GPU functions that launch threads) 
# cannot return values. We get around 
# that by passing inputs and outputs. 
# This is a common pattern in C, 
# # but not very common in Python.





@cuda.jit
def add_scalars(a, b, c):
    c[0] = a + b

dev_c = cuda.device_array((1,), np.float32)

# These square brackets refer to the number of blocks in a 
# grid, and the number of threads in a block, respectively.
add_scalars[1, 1](2.0, 7.0, dev_c)

c = dev_c.copy_to_host()
print(f"2.0 + 7.0 = {c[0]}")
#  2.0 + 7.0 = 9.0