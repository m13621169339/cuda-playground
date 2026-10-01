#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

void vectorAddCPU(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 10000000;
    size_t bytes = N * sizeof(float);

    float *a = (float*)malloc(bytes), *b = (float*)malloc(bytes);
    float *c_cpu = (float*)malloc(bytes), *c_gpu = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { a[i] = 1.0f * i; b[i] = 2.0f * i; }

    // ===== CPU 计时 =====
    auto t0 = std::chrono::high_resolution_clock::now();
    vectorAddCPU(a, b, c_cpu, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ===== GPU：分配 + 搬进 =====
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);
    auto g0 = std::chrono::high_resolution_clock::now();   // 端到端起点（含搬运）
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    // ===== GPU kernel 计时（event） =====
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);  // warmup
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernel_ms;
    cudaEventElapsedTime(&kernel_ms, start, stop);

    // ===== 搬回（计入端到端） =====
    cudaMemcpy(c_gpu, d_c, bytes, cudaMemcpyDeviceToHost);
    auto g1 = std::chrono::high_resolution_clock::now();
    double gpu_e2e_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();

    // ===== 对拍：报告最大误差 =====
    double max_err = 0;
    for (int i = 0; i < N; i++) {
        double diff = fabs((double)c_cpu[i] - (double)c_gpu[i]);
        if (diff > max_err) max_err = diff;
    }
    printf("对拍: %s, max_err = %g\n", max_err < 1e-5 ? "PASS" : "FAIL", max_err);
    printf("CPU:        %.3f ms\n", cpu_ms);
    printf("GPU kernel: %.3f ms  (kernel-only 加速 %.1f 倍)\n", kernel_ms, cpu_ms / kernel_ms);
    printf("GPU 端到端: %.3f ms  (含搬运加速 %.2f 倍)\n", gpu_e2e_ms, cpu_ms / gpu_e2e_ms);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(a); free(b); free(c_cpu); free(c_gpu);
    return 0;
}
