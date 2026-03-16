import torch
import triton
import triton.language as tl

@triton.jit
def dot_product_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    partial = tl.sum(a * b, axis=0)                # scalar partial dot product
    tl.atomic_add(out_ptr, partial)                # safely accumulate


# a, b, result are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, result: torch.Tensor, n: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)           # ceil(n / 1024) blocks
    result.zero_()                                  # must clear before accumulating
    dot_product_kernel[grid](a, b, result, n, BLOCK_SIZE=BLOCK_SIZE)
