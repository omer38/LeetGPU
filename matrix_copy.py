import torch
import triton
import triton.language as tl

@triton.jit
def matrix_copy_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * N
    x = tl.load(a_ptr + offsets, mask=mask)
    tl.store(b_ptr + offsets, x, mask=mask)

def solve(a: torch.Tensor, b: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N * N, BLOCK_SIZE),)
    matrix_copy_kernel[grid](a, b, N, BLOCK_SIZE)


def run_test(name: str, a: torch.Tensor):
    N = a.shape[0]
    assert a.shape == (N, N), f"Expected square matrix, got {a.shape}"
    a_gpu = a.cuda()
    b_gpu = torch.empty_like(a_gpu)
    solve(a_gpu, b_gpu, N)
    match = torch.allclose(b_gpu, a_gpu)
    status = "PASS" if match else "FAIL"
    print(f"[{status}] {name}")
    if not match:
        print(f"  max diff: {(b_gpu - a_gpu).abs().max().item()}")


if __name__ == "__main__":
    # Small matrix (1x1)
    run_test("1x1 matrix", torch.tensor([[5.0]]))

    # Small square matrix
    run_test("3x3 matrix", torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))

    # All zeros
    run_test("All zeros (8x8)", torch.zeros(8, 8))

    # Identity matrix
    run_test("Identity matrix (4x4)", torch.eye(4))

    # Random matrix — exercises multiple blocks
    torch.manual_seed(42)
    run_test("Random 64x64", torch.randn(64, 64))

    # Large matrix — N*N >> BLOCK_SIZE
    torch.manual_seed(123)
    run_test("Large random (128x128)", torch.randn(128, 128))

    # Non-multiple of BLOCK_SIZE — tests boundary masking (e.g. 33*33=1089)
    torch.manual_seed(0)
    run_test("Non-multiple of BLOCK_SIZE (33x33)", torch.randn(33, 33))

    # float16 (half precision)
    run_test("float16 matrix (16x16)", torch.randn(16, 16, dtype=torch.float16))