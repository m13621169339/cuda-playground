#include <cstdio>

int main() {
    float a[4] = {10, 20, 30, 40};
    for (int i = 0; i < 4; i++) {
        printf("a[%d] = %.0f, 地址 = %p\n", i, a[i], (void*)&a[i]);
    }
    printf("sizeof(float) = %zu 字节\n", sizeof(float));
    return 0;
}
