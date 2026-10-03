#include <cstdio>
#include <cstdlib>
#include <cmath>
#define TILE 16

__global__ void matmulNaive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) sum += A[row*N+k] * B[k*N+col];
        C[row*N+col] = sum;
    }
}

__global__ void matmulTiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < N / TILE; t++) {
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();
        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = sum;
}

__global__ void matmulReg(const float* A, const float* B, float* C, int N) {
    int row = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    float acc[4][4] = {0};
    for (int k = 0; k < N; k++) {
        float ra[4], rb[4];
        for (int i = 0; i < 4; i++) ra[i] = A[(row+i)*N + k];
        for (int j = 0; j < 4; j++) rb[j] = B[k*N + col + j];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                acc[i][j] += ra[i] * rb[j];
    }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            C[(row+i)*N + col+j] = acc[i][j];
}

double checkSample(const float* A, const float* B, const float* out, int N, int S) {
    double e = 0;
    for (int s = 0; s < S; s++) {
        int i = rand() % N, j = rand() % N;
        double ref = 0;
        for (int k = 0; k < N; k++) ref += (double)A[i*N+k] * B[k*N+j];
        double r = fabs(ref - out[i*N+j]) / (fabs(ref) + 1e-6);
        if (r > e) e = r;
    }
    return e;
}

int main() {
    const int N = 4096;
    size_t bytes = (size_t)N * N * sizeof(float);
    float *A = (float*)malloc(bytes), *B = (float*)malloc(bytes), *C_gpu = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N*N; i++) { A[i] = (rand()%100)/100.0f; B[i] = (rand()%100)/100.0f; }
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);
    double flops = 2.0*N*(double)N*N/1e9;

    dim3 b16(16,16), gFull((N+15)/16,(N+15)/16), gReg(N/64, N/64);
    cudaEvent_t st, sp; cudaEventCreate(&st); cudaEventCreate(&sp);
    float ms;

    matmulNaive<<<gFull, b16>>>(d_A, d_B, d_C, N); cudaDeviceSynchronize();
    cudaEventRecord(st); matmulNaive<<<gFull, b16>>>(d_A, d_B, d_C, N);
    cudaEventRecord(sp); cudaEventSynchronize(sp); cudaEventElapsedTime(&ms, st, sp);
    cudaMemcpy(C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
    printf("v0 naive:  %.2f ms, %.0f GFLOPS (对拍 %.1e)\n", ms, flops/(ms/1e3), checkSample(A,B,C_gpu,N,2000));

    matmulTiled<<<gFull, b16>>>(d_A, d_B, d_C, N); cudaDeviceSynchronize();
    cudaEventRecord(st); matmulTiled<<<gFull, b16>>>(d_A, d_B, d_C, N);
    cudaEventRecord(sp); cudaEventSynchronize(sp); cudaEventElapsedTime(&ms, st, sp);
    cudaMemcpy(C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
    printf("v2 tiling: %.2f ms, %.0f GFLOPS (对拍 %.1e)\n", ms, flops/(ms/1e3), checkSample(A,B,C_gpu,N,2000));

    matmulReg<<<gReg, b16>>>(d_A, d_B, d_C, N); cudaDeviceSynchronize();
    cudaEventRecord(st); matmulReg<<<gReg, b16>>>(d_A, d_B, d_C, N);
    cudaEventRecord(sp); cudaEventSynchronize(sp); cudaEventElapsedTime(&ms, st, sp);
    cudaMemcpy(C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
    printf("v3 寄存器: %.2f ms, %.0f GFLOPS (对拍 %.1e)\n", ms, flops/(ms/1e3), checkSample(A,B,C_gpu,N,2000));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C_gpu);
    return 0;
}
