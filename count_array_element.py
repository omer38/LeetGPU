import torch
import triton
import triton.language as tl


@triton.jit
def count_equal_kernel(input_ptr, output_ptr, N, K, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask = offs < N
    values = tl.load(input_ptr + offs, mask=mask, other=K - 1) #To fill oob with dummy value not K
    matches = (values == K)
    block_count = tl.sum(matches.to(tl.int32), axis=0)
    # Normal store — each block overwrites the output independently, (race condition: last writer wins, others are lost)
    # Atomic add — each block safely adds to the shared total, (no matter how many blocks run in parallel)
    tl.atomic_add(output_ptr, block_count)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, K: int):
    BLOCK_SIZE = 256

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    count_equal_kernel[grid](input, output, N, K, BLOCK_SIZE=BLOCK_SIZE)


def run_test(name, input_data, K):
    N = input_data.numel()
    x = input_data.cuda()
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    solve(x, out, N, K)
    expected = (input_data == K).sum().item()
    match = expected == out.item()
    status = "PASS" if match else "FAIL"
    print(f"[{status}] {name}")
    if not match:
        print(f"  expected count={expected}, got {out.item()}")


if __name__ == "__main__":
    # All elements equal to K
    run_test("All equal to K", torch.tensor([5, 5, 5, 5]), 5)

    # No elements equal to K
    run_test("None equal to K", torch.tensor([1, 2, 3, 4]), 5)

    # Mixed values
    run_test("Mixed values", torch.tensor([1, 2, 2, 3, 2, 4]), 2)

    # All zeros
    run_test("All zeros, K=0", torch.zeros(8), 0)
    run_test("All zeros, K=1", torch.zeros(8), 1)

    # Single element
    run_test("Single element match", torch.tensor([3]), 3)
    run_test("Single element no match", torch.tensor([3]), 7)

    # Large random tensor — exercises multiple blocks
    torch.manual_seed(42)
    run_test("Large random (N=100k)", torch.randint(0, 100, (100_000,)), 42)

    # Non-multiple of BLOCK_SIZE — tests boundary masking
    torch.manual_seed(123)
    run_test("Non-multiple of BLOCK_SIZE (N=1025)", torch.randint(0, 10, (1025,)), 7)
