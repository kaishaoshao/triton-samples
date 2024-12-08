import torch

import triton
import triton.language as tl


@triton.jit
def kernel(x_ptr, y_ptr, z_ptr, BLOCK_SIZE: tl.constexpr):
    tid = tl.arange(0, BLOCK_SIZE)[:, None] * \
        BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    x = tl.load(x_ptr + tid)
    y = tl.load(y_ptr + tid)
    z = tl.dot(x, y, allow_tf32=False)
    tl.store(z_ptr + tid, z)


BLOCK_SIZE = 64
x = torch.randn((BLOCK_SIZE, BLOCK_SIZE), device='cuda', dtype=torch.float32)
y = torch.randn((BLOCK_SIZE, BLOCK_SIZE), device='cuda', dtype=torch.float32)
z_triton = torch.zeros((BLOCK_SIZE, BLOCK_SIZE), device='cuda', dtype=torch.float32)
z_torch = x @ y
kernel[(1,)](x, y, z_triton, BLOCK_SIZE)
print(z_triton)
print(z_torch)