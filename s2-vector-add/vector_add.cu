#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1000000;
    size_t bytes = N * sizeof(float);

    float* a = (float*)malloc(bytes);
    float* b = (float*)malloc(bytes);
    float* c = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { a[i] = 1.0f * i; b[i] = 2.0f * i; }

    // ① 分配显存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // ② 搬进：内存 → 显存
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    // ③ 计算：发射 100 万个线程
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    // ④ 搬回：显存 → 内存
    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);

    // ⑤ 验证 + 释放
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (fabs(c[i] - 3.0f * i) > 1e-5) { ok = false; break; }
    }
    printf("%s (N=%d, grid=%d, block=%d)\n", ok ? "PASS" : "FAIL", N, gridSize, blockSize);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(a); free(b); free(c);
    return 0;
}
