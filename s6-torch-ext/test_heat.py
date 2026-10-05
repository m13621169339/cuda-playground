import torch, time
from torch.utils.cpp_extension import load

ext = load(name="heat_ext", sources=["heat_step.cu"], verbose=False)

N, T, r = 1_000_000, 1000, 0.25
u0 = torch.zeros(N, device="cuda")
u0[N//2 - 1000 : N//2 + 1000] = 100.0

# 自定义算子
torch.cuda.synchronize(); t0 = time.perf_counter()
out = ext.heat_solve(u0, T, r)
torch.cuda.synchronize(); t1 = time.perf_counter()

# 原生 torch（金标准+对照组）
u = u0.clone()
torch.cuda.synchronize(); t2 = time.perf_counter()
for _ in range(T):
    v = torch.empty_like(u)
    v[1:-1] = u[1:-1] + r * (u[:-2] - 2*u[1:-1] + u[2:])
    v[0] = u[0]; v[-1] = u[-1]
    u, v = v, u
torch.cuda.synchronize(); t3 = time.perf_counter()

err = (out - u).abs().max().item()
print(f"自定义算子: {(t1-t0)*1000:.1f} ms")
print(f"原生 torch: {(t3-t2)*1000:.1f} ms")
print(f"加速比: {(t3-t2)/(t1-t0):.1f}x")
print(f"对拍: max_err = {err:.2e}", "PASS" if err < 1e-3 else "FAIL")
