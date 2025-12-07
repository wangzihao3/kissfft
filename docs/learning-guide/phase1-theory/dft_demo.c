/*
 * DFT 演示程序
 * 展示 DFT 的基本原理和计算过程
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#define PI 3.14159265358979323846

// 朴素的 DFT 实现
void dft(double *input, double complex *output, int N) {
    printf("\n=== DFT 计算过程 ===\n");

    for (int k = 0; k < N; k++) {
        output[k] = 0;

        printf("计算 X[%d]:\n", k);

        for (int n = 0; n < N; n++) {
            double angle = -2.0 * PI * k * n / N;
            double complex twiddle = cos(angle) + I * sin(angle);

            printf("  n=%2d: x[%d] = %6.3f, W_%d^%d = %6.3f %c %6.3fi\n",
                   n, n, input[n], N, (k*n)%N,
                   creal(twiddle), cimag(twiddle) >= 0 ? '+' : '-', fabs(cimag(twiddle)));

            output[k] += input[n] * twiddle;
        }

        printf("  X[%d] = %6.3f %c %6.3fi\n\n",
               k, creal(output[k]), cimag(output[k]) >= 0 ? '+' : '-', fabs(cimag(output[k])));
    }
}

// 打印序列
void print_sequence(const char *label, double *data, int N) {
    printf("%s: [", label);
    for (int i = 0; i < N; i++) {
        printf("%6.3f", data[i]);
        if (i < N - 1) printf(", ");
    }
    printf("]\n");
}

// 打印复数序列
void print_complex_sequence(const char *label, double complex *data, int N) {
    printf("%s:\n", label);
    for (int i = 0; i < N; i++) {
        printf("  X[%d] = %8.3f %c %8.3fi\n",
               i, creal(data[i]), cimag(data[i]) >= 0 ? '+' : '-', fabs(cimag(data[i])));
    }
    printf("\n");
}

// 生成测试信号
void generate_test_signal(double *signal, int N, int type) {
    switch (type) {
        case 1: // 单一频率正弦波
            for (int i = 0; i < N; i++) {
                signal[i] = sin(2 * PI * 2 * i / N);  // 2 个周期
            }
            break;

        case 2: // 方波
            for (int i = 0; i < N; i++) {
                signal[i] = (i < N/2) ? 1.0 : -1.0;
            }
            break;

        case 3: // 两个频率分量
            for (int i = 0; i < N; i++) {
                signal[i] = sin(2 * PI * 1 * i / N) + 0.5 * sin(2 * PI * 3 * i / N);
            }
            break;

        case 4: // 脉冲
            signal[0] = 1.0;
            for (int i = 1; i < N; i++) {
                signal[i] = 0.0;
            }
            break;
    }
}

int main() {
    const int N = 8;
    double signal[N];
    double complex spectrum[N];

    printf("=== DFT 演示程序 ===\n");
    printf("序列长度 N = %d\n\n", N);

    // 测试不同类型的信号
    int signal_types[] = {1, 2, 3, 4};
    const char *signal_names[] = {
        "单一频率正弦波",
        "方波",
        "两个频率分量",
        "单位脉冲"
    };

    for (int t = 0; t < 4; t++) {
        printf("\n====================\n");
        printf("测试信号: %s\n", signal_names[t]);
        printf("====================\n");

        generate_test_signal(signal, N, signal_types[t]);
        print_sequence("输入信号 x[n]", signal, N);

        dft(signal, spectrum, N);
        print_complex_sequence("输出频谱 X[k]", spectrum, N);

        // 分析结果
        printf("频谱分析:\n");
        printf("  直流分量 (k=0):    %.3f\n", creal(spectrum[0]));
        printf("  最大幅值:         %.3f\n",
               cabs(spectrum[0]));
        for (int k = 1; k < N; k++) {
            if (cabs(spectrum[k]) > 0.1) {  // 只显示显著的频率分量
                printf("  k=%d 频率分量:      %.3f\n", k, cabs(spectrum[k]));
            }
        }
    }

    // 验证 DFT 的线性性质
    printf("\n=== 验证 DFT 线性性质 ===\n");

    double signal1[N], signal2[N], signal3[N];
    double complex spec1[N], spec2[N], spec3[N];

    // 生成两个不同的信号
    generate_test_signal(signal1, N, 1);  // 正弦波
    generate_test_signal(signal2, N, 4);  // 脉冲

    // 线性组合
    double a = 2.0, b = 0.5;
    for (int i = 0; i < N; i++) {
        signal3[i] = a * signal1[i] + b * signal2[i];
    }

    // 分别计算 DFT
    dft(signal1, spec1, N);
    dft(signal2, spec2, N);
    dft(signal3, spec3, N);

    // 验证线性性质
    printf("\n验证 DFT(a*x + b*y) = a*DFT(x) + b*DFT(y):\n");
    int success = 1;
    for (int k = 0; k < N; k++) {
        double complex expected = a * spec1[k] + b * spec2[k];
        double error = cabs(spec3[k] - expected);

        if (error > 1e-10) {
            printf("  k=%d: 错误 %.2e\n", k, error);
            success = 0;
        }
    }

    if (success) {
        printf("  ✓ 线性性质验证通过！\n");
    } else {
        printf("  ✗ 线性性质验证失败！\n");
    }

    printf("\n=== 程序结束 ===\n");
    return 0;
}