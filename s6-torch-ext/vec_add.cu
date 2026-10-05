#include <torch/extension.h>

__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

void vector_add(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int n = a.numel();
    int block = 256, grid = (n + block - 1) / block;
    vectorAddKernel<<<grid, block>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add, "CUDA vector add");
}
