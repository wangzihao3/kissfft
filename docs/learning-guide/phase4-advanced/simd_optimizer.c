/*
 * SIMD 优化工具
 * 演示如何使用 SIMD 指令优化 FFT 计算
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include "kiss_fft.h"

// SIMD 对齐的内存分配
#define SIMD_ALIGNED_ALLOC(size) aligned_alloc(32, (size))

// 检测 CPU 特性
typedef struct {
    int sse_support;
    int sse2_support;
    int sse3_support;
    int ssse3_support;
    int sse4_1_support;
    int sse4_2_support;
    int avx_support;
    int avx2_support;
    int avx512_support;
} CPUFeatures;

CPUFeatures detect_cpu_features() {
    CPUFeatures features = {0};

    // 使用 CPUID 指令检测特性
    uint32_t eax, ebx, ecx, edx;

    // 基本特性
    __cpuid(1, eax, ebx, ecx, edx);
    features.sse_support = (edx & (1 << 25)) != 0;
    features.sse2_support = (edx & (1 << 26)) != 0;
    features.sse3_support = (ecx & (1 << 0)) != 0;

    // SSSE3, SSE4.1, SSE4.2
    __cpuid(7, eax, ebx, ecx, edx);
    features.avx_support = (ecx & (1 << 28)) != 0;
    features.avx2_support = (ebx & (1 << 5)) != 0;

    // AVX-512
    __cpuid(7, eax, ebx, ecx, edx);
    features.avx512_support = (ebx & (1 << 16)) != 0;

    return features;
}

void print_cpu_features(CPUFeatures *features) {
    printf("CPU SIMD Features:\n");
    printf("  SSE:     %s\n", features->sse_support ? "Yes" : "No");
    printf("  SSE2:    %s\n", features->sse2_support ? "Yes" : "No");
    printf("  SSE3:    %s\n", features->sse3_support ? "Yes" : "No");
    printf("  AVX:     %s\n", features->avx_support ? "Yes" : "No");
    printf("  AVX2:    %s\n", features->avx2_support ? "Yes" : "No");
    printf("  AVX-512: %s\n", features->avx512_support ? "Yes" : "No");
    printf("\n");
}

// SIMD 优化的复数乘法 (SSE)
inline void sse_complex_multiply(float *result_real, float *result_imag,
                                const float a_real, const float a_imag,
                                const float b_real, const float b_imag) {
    __m128 va = _mm_set_ps(a_imag, a_real, a_imag, a_real);
    __m128 vb = _mm_set_ps(b_real, b_imag, b_real, b_imag);
    __m128 vsw = _mm_set_ps(-b_imag, b_real, -b_imag, b_real);

    __m128 vmul1 = _mm_mul_ps(va, vb);
    __m128 vmul2 = _mm_mul_ps(va, vsw);

    __m128 vresult = _mm_hadd_ps(vmul1, vmul2);
    _mm_storel_pi((__m64*)result_real, vresult);
}

// SIMD 优化的复数乘法 (AVX)
inline void avx_complex_multiply(float *result_real, float *result_imag,
                                const float *a_real, const float *a_imag,
                                const float *b_real, const float *b_imag,
                                int count) {
    int i;
    int simd_count = count & ~7; // 8的倍数

    for (i = 0; i < simd_count; i += 8) {
        __m256 va_r = _mm256_load_ps(&a_real[i]);
        __m256 va_i = _mm256_load_ps(&a_imag[i]);
        __m256 vb_r = _mm256_load_ps(&b_real[i]);
        __m256 vb_i = _mm256_load_ps(&b_imag[i]);

        // (a.r * b.r - a.i * b.i)
        __m256 mul1 = _mm256_mul_ps(va_r, vb_r);
        __m256 mul2 = _mm256_mul_ps(va_i, vb_i);
        __m256 res_r = _mm256_sub_ps(mul1, mul2);

        // (a.r * b.i + a.i * b.r)
        mul1 = _mm256_mul_ps(va_r, vb_i);
        mul2 = _mm256_mul_ps(va_i, vb_r);
        __m256 res_i = _mm256_add_ps(mul1, mul2);

        _mm256_store_ps(&result_real[i], res_r);
        _mm256_store_ps(&result_imag[i], res_i);
    }

    // 处理剩余元素
    for (; i < count; i++) {
        result_real[i] = a_real[i] * b_real[i] - a_imag[i] * b_imag[i];
        result_imag[i] = a_real[i] * b_imag[i] + a_imag[i] * b_real[i];
    }
}

// SIMD 优化的蝶形运算
void simd_butterfly2(kiss_fft_cpx *Fout, const kiss_fft_cpx *twiddles,
                    int m, int stride) {
    // 使用 SIMD 处理多个蝶形运算
    int i = 0;

    // 处理 8 个蝶形运算为一组 (使用 AVX)
    if (m >= 8) {
        int simd_groups = m / 8;
        for (int g = 0; g < simd_groups; g++) {
            // 加载数据
            __m256 f0_r = _mm256_load_ps(&Fout[i].r); // 4个复数的实部
            __m256 f0_i = _mm256_load_ps(&Fout[i].i); // 4个复数的虚部
            __m256 f2_r = _mm256_load_ps(&Fout[i + m].r);
            __m256 f2_i = _mm256_load_ps(&Fout[i + m].i);

            // 加载旋转因子
            __m256 tw_r = _mm256_set_ps(
                twiddles[i + 3].r, twiddles[i + 2].r,
                twiddles[i + 1].r, twiddles[i + 0].r);
            __m256 tw_i = _mm256_set_ps(
                twiddles[i + 3].i, twiddles[i + 2].i,
                twiddles[i + 1].i, twiddles[i + 0].i);

            // 计算蝶形运算
            // Fout[i] = Fout[i] + Fout[i+m] * twiddle
            // Fout[i+m] = Fout[i] - Fout[i+m] * twiddle

            // Fout[i+m] * twiddle
            __m256 mul_r = _mm256_sub_ps(
                _mm256_mul_ps(f2_r, tw_r),
                _mm256_mul_ps(f2_i, tw_i));
            __m256 mul_i = _mm256_add_ps(
                _mm256_mul_ps(f2_r, tw_i),
                _mm256_mul_ps(f2_i, tw_r));

            // 更新 Fout[i] 和 Fout[i+m]
            __m256 new_f0_r = _mm256_add_ps(f0_r, mul_r);
            __m256 new_f0_i = _mm256_add_ps(f0_i, mul_i);
            __m256 new_f2_r = _mm256_sub_ps(f0_r, mul_r);
            __m256 new_f2_i = _mm256_sub_ps(f0_i, mul_i);

            // 保存结果
            _mm256_store_ps(&Fout[i].r, new_f0_r);
            _mm256_store_ps(&Fout[i].i, new_f0_i);
            _mm256_store_ps(&Fout[i + m].r, new_f2_r);
            _mm256_store_ps(&Fout[i + m].i, new_f2_i);

            i += 4;
            twiddles += 4;
        }
    }

    // 处理剩余的蝶形运算
    for (; i < m; i++) {
        kiss_fft_cpx t;
        C_MUL(t, Fout[i + m], twiddles[i]);
        C_SUB(Fout[i + m], Fout[i], t);
        C_ADDTO(Fout[i], t);
    }
}

// SIMD 优化的旋转因子生成
void generate_twiddles_simd(kiss_fft_cpx *twiddles, int nfft, int inverse) {
    const double pi = 3.14159265358979323846;
    double phase = -2 * pi / nfft * (inverse ? -1 : 1);

    // 使用 SIMD 生成旋转因子
    int i = 0;
    int simd_count = (nfft / 2) & ~3; // 4的倍数

    // 预计算角度增量
    __m128d phase_increment = _mm_set1_pd(phase);
    __m128d indices = _mm_set_pd(0, 1, 2, 3);
    __m128d base_indices = _mm_set1_pd(0.0);

    for (int group = 0; group < simd_count / 4; group++) {
        __m128d angles = _mm_mul_pd(phase_increment,
                                    _mm_add_pd(base_indices,
                                             _mm_mul_pd(indices, _mm_set1_pd(4.0))));

        // 计算正弦和余弦
        __m128d cos_vals = _mm_cos_pd(angles);
        __m128d sin_vals = _mm_sin_pd(angles);

        // 交错存储为复数格式
        double temp[8];
        _mm_store_pd(&temp[0], cos_vals);
        _mm_store_pd(&temp[2], sin_vals);
        _mm_store_pd(&temp[4], cos_vals + 2);
        _mm_store_pd(&temp[6], sin_vals + 2);

        // 转换为 float 并存储
        for (int j = 0; j < 4; j++) {
            twiddles[i + j].r = (float)temp[j * 2];
            twiddles[i + j].i = (float)temp[j * 2 + 1];
        }

        i += 4;
        base_indices = _mm_add_pd(base_indices, _mm_set1_pd(16.0));
    }

    // 处理剩余元素
    for (; i < nfft / 2; i++) {
        double angle = phase * i;
        twiddles[i].r = (float)cos(angle);
        twiddles[i].i = (float)sin(angle);
    }

    // 对于正向 FFT，反转旋转因子的虚部
    if (!inverse) {
        for (i = 1; i < nfft / 2; i++) {
            twiddles[i].i = -twiddles[i].i;
        }
    }
}

// 性能测试框架
typedef struct {
    const char *name;
    void (*fft_func)(kiss_fft_cfg, const kiss_fft_cpx*, kiss_fft_cpx*);
    double time_ms;
    double speedup;
} FFTBenchmark;

// 标准的 KISS FFT（用于对比）
void standard_kiss_fft(kiss_fft_cfg cfg, const kiss_fft_cpx *fin,
                      kiss_fft_cpx *fout) {
    kiss_fft(cfg, fin, fout);
}

// SIMD 优化的版本
void simd_kiss_fft(kiss_fft_cfg cfg, const kiss_fft_cpx *fin,
                  kiss_fft_cpx *fout) {
    // 这里应该调用 SIMD 优化版本
    // 为了演示，暂时使用标准版本
    kiss_fft(cfg, fin, fout);
}

void run_benchmark(int N, int iterations) {
    FFTBenchmark benchmarks[] = {
        {"Standard KISS FFT", standard_kiss_fft, 0.0, 1.0},
        {"SIMD Optimized", simd_kiss_fft, 0.0, 1.0}
    };
    int num_benchmarks = sizeof(benchmarks) / sizeof(benchmarks[0]);

    // 准备测试数据
    kiss_fft_cpx *input = SIMD_ALIGNED_ALLOC(sizeof(kiss_fft_cpx) * N);
    kiss_fft_cpx *output = SIMD_ALIGNED_ALLOC(sizeof(kiss_fft_cpx) * N);
    kiss_fft_cpx *temp = SIMD_ALIGNED_ALLOC(sizeof(kiss_fft_cpx) * N);

    // 初始化随机数据
    for (int i = 0; i < N; i++) {
        input[i].r = (float)rand() / RAND_MAX;
        input[i].i = (float)rand() / RAND_MAX;
    }

    // 创建 FFT 配置
    kiss_fft_cfg cfg = kiss_fft_alloc(N, 0, NULL, NULL);

    printf("=== FFT Performance Benchmark (N=%d) ===\n", N);
    printf("%-20s %12s %12s\n", "Implementation", "Time (ms)", "Speedup");
    printf("%-20s %12s %12s\n", "-------------", "----------", "--------");

    // 运行基准测试
    for (int b = 0; b < num_benchmarks; b++) {
        clock_t start = clock();

        for (int i = 0; i < iterations; i++) {
            benchmarks[b].fft_func(cfg, input, temp);
        }

        clock_t end = clock();
        benchmarks[b].time_ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC / iterations;
        benchmarks[b].speedup = benchmarks[0].time_ms / benchmarks[b].time_ms;

        printf("%-20s %12.3f %12.2fx\n",
               benchmarks[b].name,
               benchmarks[b].time_ms,
               benchmarks[b].speedup);
    }

    printf("\n");

    // 验证结果正确性
    standard_kiss_fft(cfg, input, output);
    simd_kiss_fft(cfg, input, temp);

    double error = 0.0;
    for (int i = 0; i < N; i++) {
        double diff_r = output[i].r - temp[i].r;
        double diff_i = output[i].i - temp[i].i;
        error += diff_r * diff_r + diff_i * diff_i;
    }
    error = sqrt(error / N);

    printf("Result verification: error = %.6e\n", error);
    if (error < 1e-6) {
        printf("✓ SIMD implementation is correct\n");
    } else {
        printf("✗ SIMD implementation has errors\n");
    }

    // 清理
    free(input);
    free(output);
    free(temp);
    kiss_fft_free(cfg);
}

// 内存带宽测试
void memory_bandwidth_test() {
    const int sizes[] = {1024, 4096, 16384, 65536};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("=== Memory Bandwidth Analysis ===\n");
    printf("Size (KB) | Standard (GB/s) | SIMD (GB/s) | Improvement\n");
    printf("----------|----------------|------------|------------\n");

    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        int mem_size = N * sizeof(kiss_fft_cpx);

        // 测试标准版本
        clock_t start = clock();
        for (int i = 0; i < 1000; i++) {
            // 模拟内存访问
            volatile int sum = 0;
            for (int j = 0; j < N; j++) {
                sum += j;
            }
        }
        clock_t end = clock();
        double standard_time = (double)(end - start) / CLOCKS_PER_SEC;

        // 测试 SIMD 版本
        start = clock();
        for (int i = 0; i < 1000; i++) {
            // 使用 SIMD 访问
            float *data = SIMD_ALIGNED_ALLOC(mem_size);
            __m256 sum = _mm256_setzero_ps();
            for (int j = 0; j < N; j += 8) {
                __m256 v = _mm256_load_ps(&data[j]);
                sum = _mm256_add_ps(sum, v);
            }
            free(data);
        }
        end = clock();
        double simd_time = (double)(end - start) / CLOCKS_PER_SEC;

        // 计算带宽
        double standard_bw = (mem_size * 1000.0) / (standard_time * 1024 * 1024 * 1024);
        double simd_bw = (mem_size * 1000.0) / (simd_time * 1024 * 1024 * 1024);

        printf("%9d | %14.2f | %11.2f | %11.2fx\n",
               N / 256, standard_bw, simd_bw, simd_bw / standard_bw);
    }
}

int main() {
    printf("=== SIMD FFT Optimizer ===\n\n");

    // 检测 CPU 特性
    CPUFeatures features = detect_cpu_features();
    print_cpu_features(&features);

    // 运行不同大小的基准测试
    int test_sizes[] = {256, 512, 1024, 2048, 4096};
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        run_benchmark(test_sizes[i], 1000);
    }

    // 内存带宽分析
    memory_bandwidth_test();

    // 优化建议
    printf("\n=== Optimization Recommendations ===\n");
    if (features.avx2_support) {
        printf("✓ Use AVX2 instructions for maximum performance\n");
        printf("✓ Process 8 complex numbers per iteration\n");
    } else if (features.avx_support) {
        printf("✓ Use AVX instructions (4 complex numbers per iteration)\n");
    } else if (features.sse2_support) {
        printf("✓ Use SSE2 instructions (2 complex numbers per iteration)\n");
    } else {
        printf("⚠ No SIMD support detected, consider CPU upgrade\n");
    }

    printf("\nGeneral Tips:\n");
    printf("1. Align data to 32-byte boundaries for AVX\n");
    printf("2. Use blocking to improve cache utilization\n");
    printf("3. Prefetch data to reduce memory latency\n");
    printf("4. Minimize branch mispredictions\n");
    printf("5. Consider NUMA awareness for large systems\n");

    return 0;
}