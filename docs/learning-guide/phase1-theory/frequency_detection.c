#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>

#define PI 3.14159265358979323846

// 计算信号的 DFT
void compute_dft(double *signal, double complex *dft, int N) {
    for (int k = 0; k < N; k++) {
        dft[k] = 0;
        for (int n = 0; n < N; n++) {
            // 计算旋转因子: exp(-j*2*pi*k*n/N)
            double complex twiddle = cexp(-2.0 * PI * I * k * n / N);
            dft[k] += signal[n] * twiddle;
        }
    }
}

// 计算幅值谱
void compute_magnitude(double complex *dft, double *magnitude, int N) {
    for (int k = 0; k < N; k++) {
        magnitude[k] = cabs(dft[k]);
    }
}

// 频率和幅值的结构体
typedef struct {
    double frequency;
    double magnitude;
} FrequencyComponent;

// 比较函数用于排序
int compare_frequency_components(const void *a, const void *b) {
    FrequencyComponent *fa = (FrequencyComponent *)a;
    FrequencyComponent *fb = (FrequencyComponent *)b;
    if (fb->magnitude > fa->magnitude) return 1;
    if (fb->magnitude < fa->magnitude) return -1;
    return 0;
}

// 检测信号中的主要频率
void detect_frequencies(double *signal, int N, int fs) {
    // 分配内存
    double complex *dft = malloc(N * sizeof(double complex));
    double *magnitude = malloc(N * sizeof(double));
    FrequencyComponent *components = malloc(N * sizeof(FrequencyComponent));

    if (!dft || !magnitude || !components) {
        printf("内存分配失败！\n");
        free(dft);
        free(magnitude);
        free(components);
        return;
    }

    // 1. 计算 DFT
    compute_dft(signal, dft, N);

    // 2. 计算幅值谱
    compute_magnitude(dft, magnitude, N);

    // 3. 创建频率分量数组
    double freq_resolution = (double)fs / N;
    for (int k = 0; k < N; k++) {
        components[k].frequency = k * freq_resolution;
        components[k].magnitude = magnitude[k];
    }

    // 只处理前半部分频谱（奈奎斯特频率以下）
    int half_N = N / 2 + 1;
    qsort(components, half_N, sizeof(FrequencyComponent), compare_frequency_components);

    // 4. 打印结果（前 10 个最强的频率分量）
    printf("\n检测到的主要频率成分：\n");
    printf("--------------------------------\n");
    printf("%-10s %-15s %-15s\n", "排名", "频率 (Hz)", "幅值");
    printf("--------------------------------\n");

    for (int i = 0; i < 10 && i < half_N; i++) {
        printf("%-10d %-15.2f %-15.2f\n",
               i + 1,
               components[i].frequency,
               components[i].magnitude);
    }
    printf("--------------------------------\n");

    // 找出最主要的频率分量（忽略直流分量 k=0）
    printf("\n分析结果：\n");
    for (int i = 1; i < half_N; i++) {
        // 忽略幅值很小的频率分量
        if (components[i].magnitude > 0.1 * components[0].magnitude) {
            printf("检测到显著频率分量：%.2f Hz (幅值: %.2f)\n",
                   components[i].frequency,
                   components[i].magnitude);
        }
    }

    // 释放内存
    free(dft);
    free(magnitude);
    free(components);
}

int main() {
    printf("频率检测程序\n");
    printf("=================\n\n");

    // 测试信号：50 Hz + 120 Hz
    double fs = 1000;  // 采样率 1 kHz
    int N = 1024;
    double signal[N];

    printf("生成测试信号：50 Hz + 120 Hz 的正弦波组合\n");
    printf("采样率: %.0f Hz\n", fs);
    printf("采样点数: %d\n\n", N);

    // 生成测试信号
    for (int i = 0; i < N; i++) {
        double t = (double)i / fs;
        signal[i] = sin(2*PI*50*t) + 0.5*sin(2*PI*120*t);
    }

    // 检测频率
    detect_frequencies(signal, N, fs);

    printf("\n按任意键退出...\n");
    getchar();

    return 0;
}