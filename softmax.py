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


def run_test(name, input_data, atol=1e-5, rtol=1e-4):
    x = input_data.cuda()
    out = torch.empty(x.numel(), device="cuda", dtype=torch.float32)
    solve(x, out, x.numel())
    expected = torch.softmax(x.to(torch.float32), dim=0)
    match = torch.allclose(out, expected, atol=atol, rtol=rtol)
    status = "PASS" if match else "FAIL"
    print(f"[{status}] {name}")
    if not match:
        max_diff = (out - expected).abs().max().item()
        print(f"  max diff: {max_diff}")


if __name__ == "__main__":
    # Small vector
    run_test("Small vector", torch.tensor([1.0, 2.0, 3.0, 4.0]))

    # All zeros -> uniform distribution
    run_test("All zeros", torch.zeros(16))

    # Single element -> probability 1.0
    run_test("Single element", torch.tensor([5.0]))

    # Extreme values for numerical stability
    run_test("Extreme values", torch.tensor([-1e4, 0.0, 1e4]))

    # Non-multiple of typical block sizes
    torch.manual_seed(123)
    run_test("Non-multiple length (N=1025)", torch.randn(1025))

    # Large random input
    torch.manual_seed(0)
    run_test("Large random (N=1M)", torch.randn(1_000_000), atol=3e-5, rtol=3e-4)

    # Half precision input (kernel computes in fp32)
    torch.manual_seed(7)
    run_test("float16 input", torch.randn(4096, dtype=torch.float16), atol=3e-3, rtol=3e-3)
