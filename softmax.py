import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(input, output, N, BLOCK_SIZE: tl.constexpr):
    row_max = -float("inf")
    for off in tl.range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(input + cols, mask=mask, other=-float("inf")).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    denom = 0.0
    for off in tl.range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(input + cols, mask=mask, other=-float("inf")).to(tl.float32)
        denom += tl.sum(tl.exp(x - row_max), axis=0)

    for off in tl.range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(input + cols, mask=mask, other=-float("inf")).to(tl.float32)
        probs = tl.exp(x - row_max) / denom
        tl.store(output + cols, probs, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    block_size = min(1024, triton.next_power_of_2(max(1, N)))
    num_warps = 4 if block_size <= 256 else 8

    softmax_kernel[(1,)](
        input,
        output,
        N,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
