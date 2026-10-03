#include <cstdio>
#include <cstdlib>
#include <cmath>

// v0: warp 内 col 连续 → B 访问合并（好）
__global__ void matmulGood(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) sum += A[row*N+k] * B[k*N+col];
        C[row*N+col] = sum;
    }
}

// v1负对照: warp 内 row 连续 → A 访问跳跃（坏）
__global__ void matmulBad(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) sum += A[row*N+k] * B[k*N+col];
        C[row*N+col] = sum;
    }
}

typedef void (*Kernel)(const float*, const float*, float*, int);
float bench(Kernel k, const float* dA, const float* dB, float* dC, int N,
            dim3 grid, dim3 block) {
    k<<<grid, block>>>(dA, dB, dC, N);   // warmup
    cudaDeviceSynchronize();
    cudaEvent_t st, sp;
    cudaEventCreate(&st); cudaEventCreate(&sp);
    cudaEventRecord(st);
    k<<<grid, block>>>(dA, dB, dC, N);
    cudaEventRecord(sp);
    cudaEventSynchronize(sp);
    float ms; cudaEventElapsedTime(&ms, st, sp);
    return ms;
}

int main() {
    const int N = 1024;
    size_t bytes = N * N * sizeof(float);
    float *A = (float*)malloc(bytes), *B = (float*)malloc(bytes);
    float *C_cpu = (float*)malloc(bytes), *C_gpu = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N * N; i++) {
        A[i] = (rand()%100)/100.0f; B[i] = (rand()%100)/100.0f;
    }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < N; k++) s += A[i*N+k]*B[k*N+j];
            C_cpu[i*N+j] = s;
        }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16), grid((N+15)/16, (N+15)/16);
    double flops = 2.0*N*N*N/1e9;

    float ms0 = bench(matmulGood, d_A, d_B, d_C, N, grid, block);
    cudaMemcpy(C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
    double e0 = 0;
    for (int i = 0; i < N*N; i++) {
        double r = fabs(C_cpu[i]-C_gpu[i])/(fabs(C_cpu[i])+1e-6);
        if (r > e0) e0 = r;
    }

    float ms1 = bench(matmulBad, d_A, d_B, d_C, N, grid, block);
    cudaMemcpy(C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
    double e1 = 0;
    for (int i = 0; i < N*N; i++) {
        double r = fabs(C_cpu[i]-C_gpu[i])/(fabs(C_cpu[i])+1e-6);
        if (r > e1) e1 = r;
    }

    printf("v0 合并访存:   %.2f ms, %.0f GFLOPS (对拍 %.1e)\n", ms0, flops/(ms0/1e3), e0);
    printf("v1 未合并:     %.2f ms, %.0f GFLOPS (对拍 %.1e)\n", ms1, flops/(ms1/1e3), e1);
    printf("访存模式差异带来的差距: %.1f 倍\n", ms1/ms0);

    cudaFree(d_A); cudaFree(d_B); d_C && cudaFree(d_C);
    free(A); free(B); free(C_cpu); free(C_gpu);
    return 0;
}
