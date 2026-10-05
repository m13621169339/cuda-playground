# cuda-playground

从零手写 CUDA 算子的优化实验记录（RTX 4090 / CUDA 12.8）。每个实验都遵循"假设 → 实测 → 归因"的闭环。

## 核心结果：matmul 优化阶梯（N=4096, FP32）

| 版本 | GFLOPS | 相对提升 | 归因 |
|---|---|---|---|
| v0 naive | 5,087 | 1× | 合并访存+broadcast，L2/L1 接住复用 |
| v1 负对照（交换 x/y 映射） | 1,192 | 0.24× | warp 访存拼单失败：32 线程 32 事务 |
| v2 共享内存 tiling | 6,636 | 1.3× | 显存流量 ÷16，但瓶颈已转移至指令吞吐 |
| v3 寄存器分块 4×4 | 17,508 | 2.6× | load:FMA 从 2:1 降至 0.5:1 |
| v4 tiling+寄存器合体 | 36,119 | 2.1× | 双瓶颈同解，达 FP32 峰值 44% |
| cuBLAS（参照） | 57,703 | — | v4 达其 63%；差距=tensor core/双缓冲/float4/warp tiling |

## 关键实验（反向证据集）

- **合并访存负对照**：计算逻辑逐字相同的两个 kernel，仅交换线程→数据映射，性能差 4.1 倍（对拍均 PASS，3.1e-07）——性能是独立于正确性的第二战场。
- **tiling 收益实测 ≠ 教科书**：N=1024 时仅 1.3×，因为 4090 的 72MB L2 完整接住 12MB 工作集；放大到 N=4096 后 naive 依然坚挺（warp 局部性），确诊瓶颈在指令吞吐而非带宽——引出寄存器分块的 2.6×。
- **__launch_bounds__(256,6) 实验**：强制满占用率导致 96B 寄存器 spill 到 local memory，性能腰斩（35,554→17,642）——占用率是手段而非目的。
- **occupancy 手工审计**（ptxas -v）：v4 为 72 寄存器/线程 + 8KB 共享内存 → 三约束计算得 50% 占用率，寄存器受限。
- **roofline 判决**：vectorAdd AI≈0.083 FLOP/B，实测带宽 960GB/s 已达 HBM 天花板 95%，判定优化完毕；matmul 经 tiling 将 AI 抬至 ≈16 FLOP/B 进入计算区。

## 数值正确性方法论

- 全量对拍（相对误差，阈值随累加长度标定：N=1024→3.1e-7，N=4096→3.5e-6）
- 大矩阵抽样对拍（随机抽 2000 元素现场计算点积比对）
- 浮点精度边界实验：全 1 数据 ≤2^24 无舍入；改用 0.1 后非结合律效应可见

## reduction 优化阶梯（N=10^7）

原子累加 14.0ms → 共享内存树形归约 0.062ms（224×）→ warp shuffle 收尾 0.033ms（428×）

## 目录

- s1-vector-add/  C++ 工程基础（声明/实现分离、CMake、gdb）
- s2-vector-add/  首次 GPU 移植 + 端到端 benchmark（kernel 710× / e2e 2.49×）
- s2-reduction/   归约三版本 + 浮点精度实验
- s3-matmul/      matmul v0-v4 + cuBLAS 对照

## 真实数值格式移植：1D 热传导求解器（项目本体）

- 10^6 格点 × 10^3 时间步显式差分，stencil kernel + 双缓冲，数据全程驻留显存
- **加速比：CPU 1836ms → GPU 3.83ms ≈ 479×（端到端 ≈450×）**
- CPU/GPU 逐点对拍：混合容差（atol+rtol），最终误差 5.6e-45（denormal 量级，逐位一致）
- 调试实录：三轮假设定位初态污染 bug（"金标准先跑"陷阱——参照物消费掉了输入场）
- roofline 判决：AI≈0.3 FLOP/B，8MB 工作集驻留 L2（实测 ~2TB/s），判定优化收敛

## PyTorch 自定义算子

- torch.utils.cpp_extension JIT 桥：CUDA kernel 直接吃进 torch.Tensor
- stencil 算子 vs 原生 PyTorch 写法：**13.1×**（融合消除 kernel 启动开销与临时张量倒手）
