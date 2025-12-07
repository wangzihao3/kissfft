/*
 * KISS FFT 代码分析工具
 * 用于理解和调试 KISS FFT 的内部工作
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "kiss_fft.h"
#include "_kiss_fft_guts.h"

// 全局变量用于追踪
static int call_depth = 0;
static int fft_count = 0;

// 打印缩进
static void print_indent() {
    for (int i = 0; i < call_depth; i++) {
        printf("  ");
    }
}

// 分析后的打印函数
static void analyze_fft_call(kiss_fft_cfg cfg, const kiss_fft_cpx *fin,
                           kiss_fft_cpx *fout, const char *func_name) {
    print_indent();
    printf("[%s] N=%d, inverse=%d, call #%d\n",
           func_name, cfg->nfft, cfg->inverse, ++fft_count);

    // 打印输入（仅前8个）
    if (call_depth <= 2) {  // 避免打印太多
        print_indent();
        printf("  Input (first 8): ");
        for (int i = 0; i < cfg->nfft && i < 8; i++) {
            printf("(%6.3f,%6.3f) ", fin[i].r, fin[i].i);
        }
        printf("\n");
    }

    call_depth++;
}

// 分析前的打印函数
static void analyze_fft_result(kiss_fft_cfg cfg, const kiss_fft_cpx *fout,
                             const char *func_name) {
    call_depth--;

    // 打印输出（仅前8个）
    if (call_depth <= 2) {  // 避免打印太多
        print_indent();
        printf("  Output (first 8): ");
        for (int i = 0; i < cfg->nfft && i < 8; i++) {
            printf("(%6.3f,%6.3f) ", fout[i].r, fout[i].i);
        }
        printf("\n");
    }

    print_indent();
    printf("[End %s]\n\n", func_name);
}

// 打印 FFT 配置信息
void print_fft_config(kiss_fft_cfg cfg) {
    printf("=== FFT Configuration ===\n");
    printf("Nfft: %d\n", cfg->nfft);
    printf("Inverse: %d\n", cfg->inverse);
    printf("Factors: ");

    // 打印分解因子
    int *f = cfg->factors;
    for (int i = 0; i < cfg->n_factors; i++) {
        printf("%d ", f[i]);
    }
    printf("\n");

    printf("Number of factors: %d\n", cfg->n_factors);
    printf("=======================\n\n");
}

// 打印旋转因子表
void print_twiddle_factors(kiss_fft_cfg cfg) {
    printf("=== Twiddle Factors (first few) ===\n");
    kiss_fft_cpx *tw = cfg->twiddles;
    int n = cfg->nfft;

    printf("W_%d^k = e^(-j2πk/%d)\n", n, n);
    printf("k=0: (%6.3f,%6.3f)\n", tw[0].r, tw[0].i);
    printf("k=1: (%6.3f,%6.3f)\n", tw[1].r, tw[1].i);
    printf("k=2: (%6.3f,%6.3f)\n", tw[2].r, tw[2].i);
    if (n >= 8) {
        printf("k=4: (%6.3f,%6.3f)\n", tw[4].r, tw[4].i);
        printf("k=8: (%6.3f,%6.3f)\n", tw[8].r, tw[8].i);
    }
    printf("===============================\n\n");
}

// 生成测试信号
void generate_test_signal(kiss_fft_cpx *signal, int N, int signal_type) {
    switch (signal_type) {
        case 0: // 单一频率
            for (int i = 0; i < N; i++) {
                float phase = 2 * M_PI * 2 * i / N;  // 2个周期
                signal[i].r = cos(phase);
                signal[i].i = sin(phase);
            }
            break;

        case 1: // 实数信号
            for (int i = 0; i < N; i++) {
                signal[i].r = cos(2 * M_PI * 3 * i / N);  // 实数部分
                signal[i].i = 0;  // 虚部为0
            }
            break;

        case 2: // 多频率分量
            for (int i = 0; i < N; i++) {
                signal[i].r = cos(2 * M_PI * 1 * i / N) +
                             0.5 * cos(2 * M_PI * 4 * i / N);
                signal[i].i = 0;
            }
            break;

        case 3: // 脉冲
            signal[0].r = 1;
            signal[0].i = 0;
            for (int i = 1; i < N; i++) {
                signal[i].r = 0;
                signal[i].i = 0;
            }
            break;
    }
}

// 验证 FFT 结果
void verify_fft_result(kiss_fft_cpx *input, kiss_fft_cpx *output, int N, int is_inverse) {
    if (is_inverse) {
        // 对于逆FFT，输出应该接近输入
        double error = 0;
        for (int i = 0; i < N; i++) {
            double diff_r = output[i].r / N - input[i].r;  // 记得除以 N
            double diff_i = output[i].i / N - input[i].i;
            error += diff_r * diff_r + diff_i * diff_i;
        }
        error = sqrt(error / N);

        printf("Inverse FFT error: %.6f\n", error);
        if (error < 1e-6) {
            printf("✓ Inverse FFT verification passed\n");
        } else {
            printf("✗ Inverse FFT verification failed\n");
        }
    } else {
        // 对于正FFT，检查Parseval定理
        double input_energy = 0, output_energy = 0;

        for (int i = 0; i < N; i++) {
            input_energy += input[i].r * input[i].r + input[i].i * input[i].i;
            output_energy += output[i].r * output[i].r + output[i].i * output[i].i;
        }

        output_energy = output_energy / N;  // FFT 会将能量放大 N 倍

        printf("Parseval check: input=%.6f, output=%.6f\n",
               input_energy, output_energy);

        if (fabs(input_energy - output_energy) < 1e-5 * input_energy) {
            printf("✓ Parseval theorem verification passed\n");
        } else {
            printf("✗ Parseval theorem verification failed\n");
        }
    }
}

// 性能测试
void benchmark_fft(int N, int iterations) {
    kiss_fft_cfg cfg = kiss_fft_alloc(N, 0, NULL, NULL);
    kiss_fft_cpx *input = malloc(sizeof(kiss_fft_cpx) * N);
    kiss_fft_cpx *output = malloc(sizeof(kiss_fft_cpx) * N);

    // 生成随机数据
    for (int i = 0; i < N; i++) {
        input[i].r = (float)rand() / RAND_MAX;
        input[i].i = (float)rand() / RAND_MAX;
    }

    printf("\n=== Performance Test ===\n");
    printf("N=%d, iterations=%d\n", N, iterations);

    // 预热
    for (int i = 0; i < 100; i++) {
        kiss_fft(cfg, input, output);
    }

    // 计时
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        kiss_fft(cfg, input, output);
    }
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double avg_time = elapsed * 1000000 / iterations;  // 微秒

    printf("Average time: %.2f μs\n", avg_time);
    printf("Performance: %.2f Mpoints/s\n", N / avg_time);

    // 计算理论 FLOPS
    double theoretical_flops = 5 * N * log2(N) / 2;  // 复数 FFT 大约需要 5N/2 log2N 次实数运算
    printf("Theoretical FLOPs: %.0f\n", theoretical_flops);

    free(input);
    free(output);
    kiss_fft_free(cfg);
}

// 测试不同长度的 FFT
void test_different_sizes() {
    int sizes[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("\n=== Testing Different FFT Sizes ===\n");
    printf("Size   Factors                              Time (μs)\n");
    printf("-----  ----------------------------------  --------\n");

    for (int i = 0; i < num_sizes; i++) {
        int N = sizes[i];
        kiss_fft_cfg cfg = kiss_fft_alloc(N, 0, NULL, NULL);

        // 打印因子分解
        printf("%5d  ", N);
        for (int j = 0; j < cfg->n_factors; j++) {
            printf("%d ", cfg->factors[j]);
        }
        while (cfg->n_factors < 6) {
            printf("  ");
            cfg->n_factors++;
        }

        // 快速性能测试
        kiss_fft_cpx *input = malloc(sizeof(kiss_fft_cpx) * N);
        kiss_fft_cpx *output = malloc(sizeof(kiss_fft_cpx) * N);

        for (int k = 0; k < N; k++) {
            input[k].r = (float)rand() / RAND_MAX;
            input[k].i = (float)rand() / RAND_MAX;
        }

        clock_t start = clock();
        for (int j = 0; j < 1000; j++) {
            kiss_fft(cfg, input, output);
        }
        clock_t end = clock();

        double time = (double)(end - start) / CLOCKS_PER_SEC * 1000;  // 毫秒
        printf("  %8.2f\n", time);

        free(input);
        free(output);
        kiss_fft_free(cfg);
    }
}

int main() {
    printf("=== KISS FFT Code Analysis Tool ===\n\n");

    // 测试基本功能
    const int N = 16;
    kiss_fft_cfg cfg = kiss_fft_alloc(N, 0, NULL, NULL);
    kiss_fft_cpx *signal = malloc(sizeof(kiss_fft_cpx) * N);
    kiss_fft_cpx *spectrum = malloc(sizeof(kiss_fft_cpx) * N);
    kiss_fft_cpx *reconstructed = malloc(sizeof(kiss_fft_cpx) * N);

    // 打印配置信息
    print_fft_config(cfg);
    print_twiddle_factors(cfg);

    // 测试不同类型的信号
    const char *signal_names[] = {"Complex tone", "Real cosine",
                                 "Multi-tone", "Impulse"};

    for (int type = 0; type < 4; type++) {
        printf("=== Testing %s ===\n", signal_names[type]);

        // 生成测试信号
        generate_test_signal(signal, N, type);

        // 正向 FFT
        printf("\nForward FFT:\n");
        kiss_fft(cfg, signal, spectrum);

        // 逆向 FFT
        printf("\nInverse FFT:\n");
        kiss_fft_cfg icfg = kiss_fft_alloc(N, 1, NULL, NULL);
        kiss_fft(icfg, spectrum, reconstructed);

        // 验证结果
        verify_fft_result(signal, reconstructed, N, 1);

        kiss_fft_free(icfg);
        printf("\n");
    }

    // 性能测试
    benchmark_fft(1024, 10000);
    benchmark_fft(4096, 1000);

    // 测试不同大小
    test_different_sizes();

    // 清理
    free(signal);
    free(spectrum);
    free(reconstructed);
    kiss_fft_free(cfg);

    printf("\n=== Analysis Complete ===\n");
    return 0;
}