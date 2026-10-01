#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void sumAtomic(const float* x, float* sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(sum, x[i]);
}

// v2: 共享内存树形归约
__global__ void sumShared(const float* x, float* blockSums, int n) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? x[i] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) blockSums[blockIdx.x] = s[0];
}

// v3: 树形归约 + warp shuffle 收尾
__global__ void sumShuffle(const float* x, float* blockSums, int n) {
    __shared__ float s[32];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float val = (i < n) ? x[i] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    int lane = tid & 31;
    int warpId = tid >> 5;
    if (lane == 0) s[warpId] = val;
    __syncthreads();
    if (warpId == 0) {
        int numWarps = blockDim.x >> 5;
        val = (lane < numWarps) ? s[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane == 0) blockSums[blockIdx.x] = val;
    }
}

float timeKernel(void (*launch)(const float*, float*, int, int, int),
                 const float* d_x, float* d_out, int n, int grid, int block) {
    launch(d_x, d_out, n, grid, block);   // warmup
    cudaDeviceSynchronize();
    cudaEvent_t st, sp;
    cudaEventCreate(&st); cudaEventCreate(&sp);
    cudaEventRecord(st);
    launch(d_x, d_out, n, grid, block);
    cudaEventRecord(sp);
    cudaEventSynchronize(sp);
    float ms;
    cudaEventElapsedTime(&ms, st, sp);
    return ms;
}

void launchAtomic(const float* x, float* out, int n, int grid, int block) {
    cudaMemset(out, 0, sizeof(float));
    sumAtomic<<<grid, block>>>(x, out, n);
}
void launchShared(const float* x, float* out, int n, int grid, int block) {
    sumShared<<<grid, block>>>(x, out, n);
}
void launchShuffle(const float* x, float* out, int n, int grid, int block) {
    sumShuffle<<<grid, block>>>(x, out, n);
}

int main() {
    const int N = 10000000;
    size_t bytes = N * sizeof(float);
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    float* x = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) x[i] = 1.0f;
    float* partial = (float*)malloc(gridSize * sizeof(float));

    float *d_x, *d_sum, *d_partial;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_sum, sizeof(float));
    cudaMalloc(&d_partial, gridSize * sizeof(float));
    cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);

    // v1 计时
    float ms1 = timeKernel(launchAtomic, d_x, d_sum, N, gridSize, blockSize);
    float sum1;
    cudaMemcpy(&sum1, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    // v2 计时 + 汇总
    float ms2 = timeKernel(launchShared, d_x, d_partial, N, gridSize, blockSize);
    cudaMemcpy(partial, d_partial, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    double sum2 = 0;
    for (int i = 0; i < gridSize; i++) sum2 += partial[i];

    // v3 计时 + 汇总
    float ms3 = timeKernel(launchShuffle, d_x, d_partial, N, gridSize, blockSize);
    cudaMemcpy(partial, d_partial, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    double sum3 = 0;
    for (int i = 0; i < gridSize; i++) sum3 += partial[i];

    printf("真值: %d\n", N);
    printf("v1 原子累加:   %.3f ms, 结果 %.0f\n", ms1, sum1);
    printf("v2 共享内存:   %.3f ms, 结果 %.0f (相对v1加速 %.1f 倍)\n", ms2, sum2, ms1/ms2);
    printf("v3 warp洗牌:   %.3f ms, 结果 %.0f (相对v1加速 %.1f 倍)\n", ms3, sum3, ms1/ms3);

    cudaFree(d_x); cudaFree(d_sum); cudaFree(d_partial);
    free(x); free(partial);
    return 0;
}
