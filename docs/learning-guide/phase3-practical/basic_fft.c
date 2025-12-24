/**
 * basic_fft.c - 基础 KISS FFT 使用示例
 *
 * 这个示例演示了如何：
 * 1. 初始化 KISS FFT
 * 2. 准备输入数据
 * 3. 执行正向 FFT
 * 4. 执行逆向 FFT
 * 5. 验证结果
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "kiss_fft.h"

#define PI 3.14159265358979323846
#define SAMPLE_RATE 44100

// 打印复数数组
void print_complex_array(const char *label, const kiss_fft_cpx *data, int n) {
    printf("%s:\n", label);
    for (int i = 0; i < n && i < 16; i++) {
        printf("  [%2d]: %8.4f + %8.4fi\n", i, data[i].r, data[i].i);
    }
    if (n > 16) {
        printf("  ... (%d total)\n", n);
    }
}

// 生成测试信号
void generate_test_signal(kiss_fft_cpx *signal, int n, int frequency) {
    for (int i = 0; i < n; i++) {
        float t = (float)i / SAMPLE_RATE;
        signal[i].r = cosf(2 * PI * frequency * t);
        signal[i].i = sinf(2 * PI * frequency * t);
    }
}

// 计算幅值谱
void compute_magnitude_spectrum(const kiss_fft_cpx *fft_result, float *magnitude, int n) {
    for (int i = 0; i < n; i++) {
        float real = fft_result[i].r;
        float imag = fft_result[i].i;
        magnitude[i] = sqrtf(real * real + imag * imag) / n;
    }
}

// 找到峰值频率
int find_peak_frequency(const float *magnitude, int n, int sample_rate) {
    int peak_index = 1;  // 跳过直流分量（索引 0）
    float peak_value = magnitude[1];

    for (int i = 2; i < n / 2; i++) {  // 只看正频率部分
        if (magnitude[i] > peak_value) {
            peak_value = magnitude[i];
            peak_index = i;
        }
    }

    // 计算实际频率
    int frequency = peak_index * sample_rate / n;
    return frequency;
}

// 计算 MSE
double calculate_mse(const kiss_fft_cpx *original, const kiss_fft_cpx *reconstructed, int n) {
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double error_r = original[i].r - reconstructed[i].r;
        double error_i = original[i].i - reconstructed[i].i;
        mse += error_r * error_r + error_i * error_i;
    }
    return mse / n;
}

int main() {
    printf("╔════════════════════════════════════════════════╗\n");
    printf("║   KISS FFT 基础使用示例                        ║\n");
    printf("╚════════════════════════════════════════════════╝\n\n");

    // 配置参数
    const int fft_size = 1024;
    const int test_frequency = 1000;  // 1 kHz 测试信号

    printf("配置参数：\n");
    printf("  FFT 大小: %d\n", fft_size);
    printf("  采样率: %d Hz\n", SAMPLE_RATE);
    printf("  测试频率: %d Hz\n\n", test_frequency);

    // 步骤 1：分配 FFT 配置
    printf("步骤 1：初始化 FFT 配置\n");
    printf("─────────────────────────────────────────\n");

    kiss_fft_cfg fft_cfg = kiss_fft_alloc(fft_size, 0, NULL, NULL);
    kiss_fft_cfg ifft_cfg = kiss_fft_alloc(fft_size, 1, NULL, NULL);

    if (!fft_cfg || !ifft_cfg) {
        fprintf(stderr, "错误：无法分配 FFT 配置\n");
        return 1;
    }
    printf("✓ FFT 配置分配成功\n\n");

    // 步骤 2：准备输入数据
    printf("步骤 2：生成测试信号\n");
    printf("─────────────────────────────────────────\n");

    kiss_fft_cpx *input = malloc(sizeof(kiss_fft_cpx) * fft_size);
    kiss_fft_cpx *fft_output = malloc(sizeof(kiss_fft_cpx) * fft_size);
    kiss_fft_cpx *reconstructed = malloc(sizeof(kiss_fft_cpx) * fft_size);
    float *magnitude = malloc(sizeof(float) * fft_size);

    if (!input || !fft_output || !reconstructed || !magnitude) {
        fprintf(stderr, "错误：内存分配失败\n");
        return 1;
    }

    generate_test_signal(input, fft_size, test_frequency);
    print_complex_array("输入信号（前 16 个样本）", input, 16);
    printf("\n");

    // 步骤 3：执行正向 FFT
    printf("步骤 3：执行正向 FFT\n");
    printf("─────────────────────────────────────────\n");

    kiss_fft(fft_cfg, input, fft_output);
    printf("✓ FFT 计算完成\n\n");

    // 步骤 4：分析频谱
    printf("步骤 4：分析频谱\n");
    printf("─────────────────────────────────────────\n");

    compute_magnitude_spectrum(fft_output, magnitude, fft_size);

    printf("幅值谱（前 16 个频率点）：\n");
    for (int i = 0; i < 16; i++) {
        printf("  Bin %2d (%5d Hz): %.6f\n", i, i * SAMPLE_RATE / fft_size, magnitude[i]);
    }

    int detected_frequency = find_peak_frequency(magnitude, fft_size, SAMPLE_RATE);
    printf("\n检测到的峰值频率: %d Hz", detected_frequency);
    printf("（预期: %d Hz，误差: %d Hz）\n\n",
           test_frequency, abs(detected_frequency - test_frequency));

    // 步骤 5：执行逆向 FFT
    printf("步骤 5：执行逆向 FFT（重建信号）\n");
    printf("─────────────────────────────────────────\n");

    kiss_fft(ifft_cfg, fft_output, reconstructed);

    // 归一化
    for (int i = 0; i < fft_size; i++) {
        reconstructed[i].r /= fft_size;
        reconstructed[i].i /= fft_size;
    }

    printf("✓ IFFT 计算完成\n\n");

    print_complex_array("重建信号（前 16 个样本）", reconstructed, 16);
    printf("\n");

    // 步骤 6：验证结果
    printf("步骤 6：验证结果\n");
    printf("─────────────────────────────────────────\n");

    double mse = calculate_mse(input, reconstructed, fft_size);
    double rmse = sqrt(mse);

    printf("均方误差 (MSE): %.10f\n", mse);
    printf("均方根误差 (RMSE): %.10f\n", rmse);

    // 计算信噪比
    double signal_power = 0;
    for (int i = 0; i < fft_size; i++) {
        signal_power += input[i].r * input[i].r + input[i].i * input[i].i;
    }
    double snr = 10 * log10(signal_power / mse / fft_size);
    printf("信噪比 (SNR): %.2f dB\n\n", snr);

    if (rmse < 0.0001) {
        printf("✓ 验证通过！重建信号与原始信号匹配。\n");
    } else {
        printf("✗ 警告：重建误差较大。\n");
    }
    printf("\n");

    // 清理资源
    printf("清理资源...\n");
    free(input);
    free(fft_output);
    free(reconstructed);
    free(magnitude);
    kiss_fft_free(fft_cfg);
    kiss_fft_free(ifft_cfg);
    printf("✓ 完成\n\n");

    printf("╔════════════════════════════════════════════════╗\n");
    printf("║   示例程序成功完成！                            ║\n");
    printf("╚════════════════════════════════════════════════╝\n");

    return 0;
}
