#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

int main() {
    const int N = 1000000;      // 一百万个格点
    const int T = 1000;         // 一千个时间步
    const float r = 0.25f;      // r = α·dt/dx²，稳定条件 r <= 0.5
    float* u = (float*)malloc(N * sizeof(float));
    float* v = (float*)malloc(N * sizeof(float));

    // 初始条件：棒中央一个 100 度的热斑，其余 0 度
    for (int i = 0; i < N; i++) u[i] = 0.0f;
    for (int i = N/2 - 1000; i < N/2 + 1000; i++) u[i] = 100.0f;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < T; t++) {
        for (int i = 1; i < N - 1; i++)              // stencil 主循环
            v[i] = u[i] + r * (u[i-1] - 2.0f*u[i] + u[i+1]);
        v[0] = u[0]; v[N-1] = u[N-1];                // 边界：两端温度固定
        float* tmp = u; u = v; v = tmp;              // 新旧场交换（指针交换，零拷贝）
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 健全性检查：热斑应扩散（中央降温、两侧升温）、无 NaN
    double total = 0, center = u[N/2];
    int bad = 0;
    for (int i = 0; i < N; i++) { total += u[i]; if (std::isnan(u[i]) || std::isinf(u[i])) bad++; }
    printf("CPU 耗时: %.1f ms\n", ms);
    printf("中央温度: %.2f (初始100，应明显下降)\n", center);
    printf("总热量: %.1f (应≈200000，扩散不灭失)\n", total);
    printf("NaN/Inf 格点数: %d (应为0，否则发散)\n", bad);
    free(u); free(v);
    return 0;
}
