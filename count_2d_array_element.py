import torch
import triton
import triton.language as tl

@triton.jit
def count_eq_kernel(inp_ptr, out_ptr, total_elems, K: tl.int32,
                    BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems

    x = tl.load(inp_ptr + offs, mask=mask, other=0).to(tl.int32)
    is_eq = x == K
    cnt = tl.sum(is_eq.to(tl.int32), axis=0)

    tl.atomic_add(out_ptr, cnt)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, M: int, K: int):
    # Flatten view
    inp = input.reshape(-1)
    total = N * M

    # We need to ensure that output is a single int32 value on GPU
    output.zero_()

    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)

    count_eq_kernel[grid](inp, output, total, K, BLOCK=BLOCK, num_warps=4,)


def run_test(name, input_data, K):
    N, M = input_data.shape
    x = input_data.cuda()
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    solve(x, out, N, M, K)
    expected = (input_data == K).sum().item()
    match = expected == out.item()
    status = "PASS" if match else "FAIL"
    print(f"[{status}] {name}")
    if not match:
        print(f"  expected count={expected}, got {out.item()}")


if __name__ == "__main__":
    # All elements equal to K
    run_test("All equal to K", torch.tensor([[5, 5], [5, 5]]), 5)

    # No elements equal to K
    run_test("None equal to K", torch.tensor([[1, 2], [3, 4]]), 5)

    # Mixed values
    run_test("Mixed values", torch.tensor([[1, 2, 2], [3, 2, 4]]), 2)

    # All zeros
    run_test("All zeros, K=0", torch.zeros(4, 4), 0)
    run_test("All zeros, K=1", torch.zeros(4, 4), 1)

    # Single element (1x1)
    run_test("Single element match", torch.tensor([[3]]), 3)
    run_test("Single element no match", torch.tensor([[3]]), 7)

    # Large random — exercises multiple blocks
    torch.manual_seed(42)
    run_test("Large random (100x100)", torch.randint(0, 100, (100, 100)), 42)

    # Non-multiple of BLOCK_SIZE — tests boundary masking
    torch.manual_seed(123)
    run_test("Non-multiple of BLOCK_SIZE (33x33)", torch.randint(0, 10, (33, 33)), 7)