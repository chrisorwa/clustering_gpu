

# load required libraries
import time 
import numpy as np
import pyopencl as cl

t0 = time.time()

# set-up GPU
platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=my_gpu_devices)
print my_gpu_devices

# generate variables
a_np = np.random.rand(5000000).astype(np.float32)
b_np = np.random.rand(5000000).astype(np.float32)

queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)


prg = cl.Program(ctx, """

#include <string.h>
#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))

 
int levenshtein(char *s1, char *s2) {
    unsigned int x, y, s1len, s2len;
    s1len = strlen(s1);
    s2len = strlen(s2);
    unsigned int matrix[s2len+1][s1len+1];
    matrix[0][0] = 0;
    for (x = 1; x <= s2len; x++)
        matrix[x][0] = matrix[x-1][0] + 1;
    for (y = 1; y <= s1len; y++)
        matrix[0][y] = matrix[0][y-1] + 1;
    for (x = 1; x <= s2len; x++)
        for (y = 1; y <= s1len; y++)
            matrix[x][y] = MIN3(matrix[x-1][y] + 1, matrix[x][y-1] + 1, matrix[x-1][y-1] + (s1[y-1] == s2[x-1] ? 0 : 1));
 
    return(matrix[s2len][s1len]);
}

__kernel void sum(__global const char *a_g,__global char *b_g, __global float *res_g) {
  int gid = get_global_id(0);
  res_g[gid] = levenshtein(a_g[gid],b_g[gid]);
  
}
""").build()

res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
prg.sum(queue, a_np.shape, None, a_g, b_g,res_g)

res_np = np.empty_like(a_np)
cl.enqueue_copy(queue, res_np, res_g)


# print variables
t = time.time() - t0

print  t
