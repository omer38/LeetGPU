import torch
import triton
import triton.language as tl


@triton.jit
def leaky_relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(input_ptr+offs,mask=mask)
    y = tl.where(x>0,x,x*0.01)
    tl.store(output_ptr+offs,y,mask=mask)
    

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    leaky_relu_kernel[grid](input, output, N, BLOCK_SIZE)


def run_test(name, input_data, negative_slope=0.01):
    x = input_data.cuda()
    out = torch.empty_like(x)
    solve(x, out, x.numel())
    expected = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
    match = torch.allclose(out, expected)
    status = "PASS" if match else "FAIL"
    print(f"[{status}] {name}")
    if not match:
        print(f"  max diff: {(out - expected).abs().max().item()}")


if __name__ == "__main__":
    # Basic positive values — all should pass through unchanged
    run_test("All positive", torch.tensor([1.0, 2.0, 3.0, 4.0]))

    # Basic negative values — all should become x * 0.01
    run_test("All negative", torch.tensor([-1.0, -2.0, -3.0, -4.0]))

    # Mixed positive and negative
    run_test("Mixed values", torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0]))

    # Zero input — boundary: LeakyReLU(0) = 0
    run_test("All zeros", torch.zeros(8))

    # Single element
    run_test("Single positive element", torch.tensor([5.0]))
    run_test("Single negative element", torch.tensor([-5.0]))

    # Large random tensor — exercises multiple blocks (N >> BLOCK_SIZE)
    torch.manual_seed(0)
    run_test("Large random tensor (N=1M)", torch.randn(1_000_000))

    # Non-multiple of BLOCK_SIZE — tests boundary masking
    run_test("Non-multiple of BLOCK_SIZE (N=1025)", torch.randn(1025))

    # Extreme values — checks numerical stability
    run_test("Extreme values", torch.tensor([-1e30, -1.0, 0.0, 1.0, 1e30]))

    # float16 (half precision)
    run_test("float16 tensor", torch.randn(4096, dtype=torch.float16))