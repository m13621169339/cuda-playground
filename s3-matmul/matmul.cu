#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void matmulNaive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * N * sizeof(float);
    float *A = (float*)malloc(bytes), *B = (float*)malloc(bytes);
    float *C_cpu = (float*)malloc(bytes), *C_gpu = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N * N; i++) {
        A[i] = (rand() % 100) / 100.0f;
        B[i] = (rand() % 100) / 100.0f;
    }

    // CPU 基准（要跑几秒，正常）
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < N; k++) s += A[i*N+k] * B[k*N+j];
            C_cpu[i*N+j] = s;
        }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    // warmup + event 计时（老规矩）
    matmulNaive<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaEvent_t st, sp;
    cudaEventCreate(&st); cudaEventCreate(&sp);
    cudaEventRecord(st);
    matmulNaive<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(sp);
    cudaEventSynchronize(sp);
    float ms;
    cudaEventElapsedTime(&ms, st, sp);

    cudaMemcpy(C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);

    // 对拍：matmul 累加 N 次，误差随 N 增长，用相对误差判
    double max_rel = 0;
    for (int i = 0; i < N * N; i++) {
        double diff = fabs(C_cpu[i] - C_gpu[i]);
        double rel = diff / (fabs(C_cpu[i]) + 1e-6);
        if (rel > max_rel) max_rel = rel;
    }

    double gflops = 2.0 * N * N * N / (ms / 1000.0) / 1e9;
    printf("对拍: max_rel_err = %.2e %s\n", max_rel, max_rel < 1e-4 ? "PASS" : "FAIL");
    printf("v0 naive: %.2f ms, %.0f GFLOPS\n", ms, gflops);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C_cpu); free(C_gpu);
    return 0;
}
