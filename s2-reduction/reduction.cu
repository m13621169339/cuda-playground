#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void sumAtomic(const float* x, float* sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(sum, x[i]);
}

int main() {
    const int N = 10000000;
    size_t bytes = N * sizeof(float);

    float* x = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) x[i] = 1.0f;   // 全填1，真值=N，心算可验

    // CPU 基准
    double cpu_sum = 0;
    for (int i = 0; i < N; i++) cpu_sum += x[i];

    // GPU v1: 原子累加
    float *d_x, *d_sum;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);

    float zero = 0.0f;
    cudaMemcpy(d_sum, &zero, sizeof(float), cudaMemcpyHostToDevice);  // 累加器清零！

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    sumAtomic<<<gridSize, blockSize>>>(d_x, d_sum, N);
    cudaDeviceSynchronize();

    float gpu_sum;
    cudaMemcpy(&gpu_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    printf("真值:   %d\n", N);
    printf("CPU:    %.1f\n", cpu_sum);
    printf("GPU v1: %.1f (误差 %.6f)\n", gpu_sum, fabs(gpu_sum - (float)N));
    printf("CPU vs GPU 相对误差: %.2e\n", fabs(cpu_sum - gpu_sum) / N);

    cudaFree(d_x); cudaFree(d_sum);
    free(x);
    return 0;
}
