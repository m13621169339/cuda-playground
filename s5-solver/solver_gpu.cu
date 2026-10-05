#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

__global__ void heatStep(const float* u, float* v, int N, float r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < N - 1)
        v[i] = u[i] + r * (u[i-1] - 2.0f*u[i] + u[i+1]);
}

int main() {
    const int N = 1000000, T = 1000;
    const float r = 0.25f;
    size_t bytes = N * sizeof(float);

    float* u = (float*)malloc(bytes);
    float* u_cpu = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) u[i] = 0.0f;
    for (int i = N/2 - 1000; i < N/2 + 1000; i++) u[i] = 100.0f;

    // CPU 金标准（同步算出用于对拍）
    float* v = (float*)malloc(bytes);
    float* a = u, *b = v;
    auto c0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < T; t++) {
        for (int i = 1; i < N - 1; i++)
            b[i] = a[i] + r * (a[i-1] - 2.0f*a[i] + a[i+1]);
        b[0] = a[0]; b[N-1] = a[N-1];
        std::swap(a, b);
    }
    auto c1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
    memcpy(u_cpu, a, bytes);

    // 关键：重新填初始条件——CPU 金标准的指针交换已把 u 写成了终态
    for (int i = 0; i < N; i++) u[i] = 0.0f;
    for (int i = N/2 - 1000; i < N/2 + 1000; i++) u[i] = 100.0f;

    // GPU 版：五段式，数据驻留
    float *d_u, *d_v;
    cudaMalloc(&d_u, bytes);
    cudaMalloc(&d_v, bytes);
    cudaMemcpy(d_u, u, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, u, bytes, cudaMemcpyHostToDevice);   // v 边界 = u 边界

    int block = 256, grid = (N + block - 1) / block;
    cudaEvent_t st, sp;
    cudaEventCreate(&st); cudaEventCreate(&sp);
    heatStep<<<grid, block>>>(d_u, d_v, N, r);   // warmup（会推进一步！）
    cudaDeviceSynchronize();
    cudaMemcpy(d_u, u, bytes, cudaMemcpyHostToDevice);  // warmup 后重置物理场
    cudaMemcpy(d_v, u, bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(st);
    for (int t = 0; t < T; t++) {
        heatStep<<<grid, block>>>(d_u, d_v, N, r);
        std::swap(d_u, d_v);
    }
    cudaEventRecord(sp);
    cudaEventSynchronize(sp);
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, st, sp);

    float* u_gpu = (float*)malloc(bytes);
    cudaMemcpy(u_gpu, d_u, bytes, cudaMemcpyDeviceToHost);

    // 对拍：绝对+相对双指标，各自定位
    double max_abs = 0, max_rel = 0;
    int ia = -1, ir = -1;
    for (int i = 0; i < N; i++) {
        double diff = fabs(u_cpu[i] - u_gpu[i]);
        if (diff > max_abs) { max_abs = diff; ia = i; }
        double rel = diff / (fabs(u_cpu[i]) + 1e-6);
        if (rel > max_rel) { max_rel = rel; ir = i; }
    }
    printf("CPU: %.1f ms\n", cpu_ms);
    printf("GPU: %.2f ms (1000步, kernel-only)\n", gpu_ms);
    printf("加速比: %.1f 倍\n", cpu_ms / gpu_ms);
    printf("max_abs = %.3e @%d (cpu=%.6f gpu=%.6f)\n", max_abs, ia, u_cpu[ia], u_gpu[ia]);
    printf("max_rel = %.3e @%d (cpu=%.6f gpu=%.6f)\n", max_rel, ir, u_cpu[ir], u_gpu[ir]);
    printf("对拍: %s (绝对容差 1e-3)\n", max_abs < 1e-3 ? "PASS" : "FAIL");

    cudaFree(d_u); cudaFree(d_v);
    free(u); free(v); free(u_cpu); free(u_gpu);
    return 0;
}
