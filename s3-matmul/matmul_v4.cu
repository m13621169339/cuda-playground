#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>

// v4: 共享内存tiling + 寄存器4x4 合体。block=256线程算64x64
__global__ void matmulV4(const float* A, const float* B, float* C, int N) {
    __shared__ float As[64][16];
    __shared__ float Bs[16][64];
    int tid = threadIdx.y * 16 + threadIdx.x;
    int row0 = blockIdx.y * 64 + threadIdx.y * 4;
    int col0 = blockIdx.x * 64 + threadIdx.x * 4;
    float acc[4][4] = {0};
    for (int t = 0; t < N / 16; t++) {
        for (int i = 0; i < 4; i++) {
            int idx = tid + i * 256;
            As[idx / 16][idx % 16] = A[(blockIdx.y*64 + idx/16) * N + t*16 + idx%16];
            Bs[idx / 64][idx % 64] = B[(t*16 + idx/64) * N + blockIdx.x*64 + idx%64];
        }
        __syncthreads();
        for (int kk = 0; kk < 16; kk++) {
            float ra[4], rb[4];
            for (int i = 0; i < 4; i++) ra[i] = As[threadIdx.y*4 + i][kk];
            for (int j = 0; j < 4; j++) rb[j] = Bs[kk][threadIdx.x*4 + j];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    acc[i][j] += ra[i] * rb[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            C[(row0+i)*N + col0+j] = acc[i][j];
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
    cudaEvent_t st, sp; cudaEventCreate(&st); cudaEventCreate(&sp);
    float ms;

    // v4 手写合体
    dim3 block(16,16), grid(N/64, N/64);
    matmulV4<<<grid, block>>>(d_A, d_B, d_C, N); cudaDeviceSynchronize();
    cudaEventRecord(st); matmulV4<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(sp); cudaEventSynchronize(sp); cudaEventElapsedTime(&ms, st, sp);
    cudaMemcpy(C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
    printf("v4 合体:   %.2f ms, %.0f GFLOPS (对拍 %.1e)\n", ms, flops/(ms/1e3), checkSample(A,B,C_gpu,N,2000));

    // cuBLAS 裁判（列主序库，传 B,A 利用转置恒等式算出行主序 A*B）
    cublasHandle_t h; cublasCreate(&h);
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaDeviceSynchronize();
    cudaEventRecord(st);
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaEventRecord(sp); cudaEventSynchronize(sp); cudaEventElapsedTime(&ms, st, sp);
    cudaMemcpy(C_gpu, d_C, bytes, cudaMemcpyDeviceToHost);
    printf("cuBLAS:    %.2f ms, %.0f GFLOPS (对拍 %.1e)\n", ms, flops/(ms/1e3), checkSample(A,B,C_gpu,N,2000));
    cublasDestroy(h);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C_gpu);
    return 0;
}
