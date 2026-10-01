#include <cstdio>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("设备名: %s\n", prop.name);
    printf("SM 数量: %d\n", prop.multiProcessorCount);
    printf("显存总量: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("每 block 最大线程数: %d\n", prop.maxThreadsPerBlock);
    printf("每 SM 共享内存: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    return 0;
}
