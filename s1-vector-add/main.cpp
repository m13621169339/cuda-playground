#include <cstdio>
#include <cstdlib>
#include "vector_add.h"

int main() {
    const int N = 1000;

    float* a = (float*)malloc(N * sizeof(float));
    float* b = (float*)malloc(N * sizeof(float));
    float* c = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f * i;
        b[i] = 2.0f * i;
    }

    vectorAdd(a, b, c, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (c[i] != 3.0f * i) { ok = false; break; }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    free(a); free(b); free(c);
    return 0;
}
