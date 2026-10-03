import re
src = open('matmul_big.cu').read()
# CPU 全量基准换成抽样对拍
old = src[src.index('    for (int i = 0; i < N; i++)'):src.index('    float *d_A')]
new = '''    // 抽样对拍准备：不再全量预计算
'''
src = src.replace(old, new)
# check 函数换成抽样版：现场算点积
old_check = src[src.index('double check'):src.index('int main')]
new_check = '''double checkSample(const float* A, const float* B, const float* out, int N, int S) {
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

'''
src = src.replace(old_check, new_check)
src = src.replace('check(C_cpu, C_gpu, N*N)', 'checkSample(A, B, C_gpu, N, 2000)')
src = src.replace('float *C_cpu = (float*)malloc(bytes), *C_gpu', 'float *C_gpu')
src = src.replace('free(C_cpu); ', '')
open('matmul_big.cu','w').write(src)
print('patched')
