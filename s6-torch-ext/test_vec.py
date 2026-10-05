import torch
from torch.utils.cpp_extension import load

ext = load(name="vec_add_ext", sources=["vec_add.cu"], verbose=False)

n = 10_000_000
a = torch.randn(n, device="cuda")
b = torch.randn(n, device="cuda")
c = torch.empty_like(a)

ext.vector_add(a, b, c)
torch.cuda.synchronize()

max_err = (c - (a + b)).abs().max().item()
print(f"对拍: max_err = {max_err:.2e}", "PASS" if max_err == 0 else "CHECK")
print("前 5 个元素:", c[:5].tolist())
