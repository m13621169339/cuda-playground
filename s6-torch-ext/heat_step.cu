#include <torch/extension.h>

__global__ void heatStepKernel(const float* u, float* v, int N, float r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < N - 1)
        v[i] = u[i] + r * (u[i-1] - 2.0f*u[i] + u[i+1]);
}

torch::Tensor heat_solve(torch::Tensor u0, int64_t T, double r) {
    auto u = u0.clone();            // 不污染调用者输入（E19 的教训）
    auto v = u0.clone();            // 边界=固定边界条件
    int N = u.numel();
    int block = 256, grid = (N + block - 1) / block;
    float rf = (float)r;
    for (int t = 0; t < T; t++) {
        heatStepKernel<<<grid, block>>>(
            u.data_ptr<float>(), v.data_ptr<float>(), N, rf);
        std::swap(u, v);            // 句柄交换，零拷贝
    }
    return u;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("heat_solve", &heat_solve, "1D heat solver, T steps on GPU");
}
