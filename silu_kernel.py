import torch
import triton
import triton.language as tl


@triton.jit
def silu_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(input + offs, mask=mask)
    x_f32 = x.to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-x_f32))
    out = x_f32 * sigmoid
    tl.store(output + offs, out, mask = mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    silu_kernel[grid](input, output, N, BLOCK_SIZE)


def run_test(name, input_data, atol=1e-5, rtol=1e-4):
    x = input_data.cuda()
    out = torch.empty(x.numel(), device="cuda", dtype=torch.float32)
    solve(x, out, x.numel())
    expected = torch.nn.functional.silu(x.to(torch.float32))
    match = torch.allclose(out, expected, atol=atol, rtol=rtol)
    status = "PASS" if match else "FAIL"
    print(f"[{status}] {name}")
    if not match:
        max_diff = (out - expected).abs().max().item()
        print(f"  max diff: {max_diff}")


if __name__ == "__main__":
    # Small vector
    run_test("Small vector", torch.tensor([1.0, 2.0, 3.0, 4.0]))

    # All zeros -> SiLU(0) = 0
    run_test("All zeros", torch.zeros(16))

    # Single element
    run_test("Single element", torch.tensor([5.0]))

    # Extreme values
    run_test("Extreme values", torch.tensor([-1e4, 0.0, 1e4]))

    # Non-multiple of typical block sizes
    torch.manual_seed(123)
    run_test("Non-multiple length (N=1025)", torch.randn(1025))

    # Large random input
    torch.manual_seed(0)
    run_test("Large random (N=1M)", torch.randn(1_000_000), atol=3e-5, rtol=3e-4)

    # Half precision input
    torch.manual_seed(7)
    run_test("float16 input", torch.randn(4096, dtype=torch.float16), atol=3e-3, rtol=3e-3)
