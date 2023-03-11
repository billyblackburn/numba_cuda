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

N = 20
a = np.arange(N, dtype=np.float32)
b = np.arange(N, dtype=np.float32)

dev_a = cuda.to_device(a) #host array moved to gpu
dev_b = cuda.to_device(b)

dev_c = cuda.device_array_like(a)

add_array[4, 8](dev_a, dev_b, dev_c)
# NumbaPerformanceWarning: Host array used in CUDA kernel will 
# incur copy overhead to/from device. i.e a and b are not on gpu
print(a)
print(b)
# [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10.
#  11. 12. 13. 14. 15. 16. 17. 18. 19.]

print(dev_c)# <numba.cuda.cudadrv.devicearray.DeviceNDArray 
# object at 0x7f9dcad9d8b0>

c = dev_c.copy_to_host()
print(c)

#  [ 0.  2.  4.  6.  8. 10. 12. 14. 16. 18. 20. 
#    22. 24. 26. 28. 30. 32. 34. 36. 38.]