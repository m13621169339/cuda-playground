#include <cstdio>

__global__ void hello() {
    printf("Hello from GPU! block=%d thread=%d\n", blockIdx.x, threadIdx.x);
}

int main() {
    hello<<<2, 4>>>();
    cudaDeviceSynchronize();
    printf("Hello from CPU!\n");
    return 0;
}
