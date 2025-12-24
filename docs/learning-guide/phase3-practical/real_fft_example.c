/**
 * real_fft_example.c - 实数 FFT 使用示例
 *
 * 这个示例演示了如何使用 KISS FFT 的实数 FFT (kiss_fftr)
 * 实数 FFT 优化了输入为实数的情况，比复数 FFT 更快
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "kiss_fftr.h"

#define PI 3.14159265358979323846
#define SAMPLE_RATE 44100

// 打印实数数组
void print_real_array(const char *label, const float *data, int n) {
    printf("%s:\n", label);
    for (int i = 0; i < n && i < 16; i++) {
        printf("  [%2d]: %8.4f\n", i, data[i]);
    }
    if (n > 16) {
        printf("  ... (%d total)\n", n);
    }
}

// 打印复数数组（实数 FFT 结果）
void print_fft_result(const kiss_fft_cpx *fft_result, int nfft) {
    int n_bins = nfft / 2 + 1;
    printf("FFT 结果（前 16 个频率点）：\n");
    printf("  Bin  |  频率 (Hz)  |   实部    |   虚部    |   幅值   \n");
    printf("───────┼─────────────┼───────────┼───────────┼──────────\n");

    for (int i = 0; i < 16 && i < n_bins; i++) {
        float freq = (float)i * SAMPLE_RATE / nfft;
        float magnitude = sqrtf(fft_result[i].r * fft_result[i].r +
                                fft_result[i].i * fft_result[i].i) / nfft;
        printf("  %2d   |  %6.0f     | %9.4f | %9.4f | %8.6f\n",
               i, freq, fft_result[i].r, fft_result[i].i, magnitude);
    }

    if (n_bins > 16) {
        printf("  ... (%d total bins)\n", n_bins);
    }
    printf("\n");
}

// 生成多频率测试信号
void generate_multi_tone_signal(float *signal, int n, const int *frequencies, int num_freqs) {
    memset(signal, 0, sizeof(float) * n);

    for (int f = 0; f < num_freqs; f++) {
        for (int i = 0; i < n; i++) {
            float t = (float)i / SAMPLE_RATE;
            signal[i] += cosf(2 * PI * frequencies[f] * t);
        }
    }

    // 归一化
    float max_val = 0;
    for (int i = 0; i < n; i++) {
        if (fabsf(signal[i]) > max_val) {
            max_val = fabsf(signal[i]);
        }
    }
    if (max_val > 0) {
        for (int i = 0; i < n; i++) {
            signal[i] /= max_val;
        }
    }
}

// 应用汉明窗
void apply_hamming_window(float *signal, int n) {
    for (int i = 0; i < n; i++) {
        float window = 0.54f - 0.46f * cosf(2 * PI * i / (n - 1));
        signal[i] *= window;
    }
}

// 分析频谱
void analyze_spectrum(const kiss_fft_cpx *fft_result, int nfft,
                     float *peak_freq, float *peak_magnitude) {
    int n_bins = nfft / 2 + 1;
    int peak_index = 1;
    float max_magnitude = 0;

    for (int i = 1; i < n_bins; i++) {
        float magnitude = sqrtf(fft_result[i].r * fft_result[i].r +
                                fft_result[i].i * fft_result[i].i);
        if (magnitude > max_magnitude) {
            max_magnitude = magnitude;
            peak_index = i;
        }
    }

    *peak_freq = (float)peak_index * SAMPLE_RATE / nfft;
    *peak_magnitude = max_magnitude / nfft;
}

// 计算功率谱密度
void compute_power_spectrum(const kiss_fft_cpx *fft_result, float *power, int nfft) {
    int n_bins = nfft / 2 + 1;
    for (int i = 0; i < n_bins; i++) {
        float real = fft_result[i].r;
        float imag = fft_result[i].i;
        power[i] = (real * real + imag * imag) / (nfft * nfft);
    }
}

// 打印频谱图（文本模式）
void print_spectrum_bars(const float *magnitude, int n_bins, float threshold) {
    const char bars[] = " _▁▂▃▄▅▆▇█";
    const int num_levels = sizeof(bars) - 1;

    printf("\n频谱图（归一化，阈值=%.3f）：\n", threshold);
    for (int i = 1; i < n_bins && i < 32; i++) {  // 跳过直流分量
        if (magnitude[i] > threshold) {
            int level = (int)(magnitude[i] * num_levels);
            if (level >= num_levels) level = num_levels - 1;
            printf("%c", bars[level]);
        } else {
            printf(" ");
        }
    }
    printf("\n");
}

int main() {
    printf("╔════════════════════════════════════════════════╗\n");
    printf("║   KISS FFT 实数 FFT 使用示例                   ║\n");
    printf("╚════════════════════════════════════════════════╝\n\n");

    // 配置参数
    const int fft_size = 1024;
    const int frequencies[] = {440, 1000, 2500};  // A4, 1kHz, 2.5kHz
    const int num_freqs = sizeof(frequencies) / sizeof(frequencies[0]);

    printf("配置参数：\n");
    printf("  FFT 大小: %d\n", fft_size);
    printf("  采样率: %d Hz\n", SAMPLE_RATE);
    printf("  测试频率: ");
    for (int i = 0; i < num_freqs; i++) {
        printf("%d Hz", frequencies[i]);
        if (i < num_freqs - 1) printf(", ");
    }
    printf("\n\n");

    // 步骤 1：分配实数 FFT 配置
    printf("步骤 1：初始化实数 FFT 配置\n");
    printf("─────────────────────────────────────────\n");

    kiss_fftr_cfg fft_cfg = kiss_fftr_alloc(fft_size, 0, NULL, NULL);
    kiss_fftr_cfg ifft_cfg = kiss_fftr_alloc(fft_size, 1, NULL, NULL);

    if (!fft_cfg || !ifft_cfg) {
        fprintf(stderr, "错误：无法分配 FFT 配置\n");
        return 1;
    }
    printf("✓ 实数 FFT 配置分配成功\n");
    printf("  频率点数: %d（实数 FFT 只需要 N/2+1 个复数）\n\n", fft_size / 2 + 1);

    // 步骤 2：生成实数测试信号
    printf("步骤 2：生成多频率测试信号\n");
    printf("─────────────────────────────────────────\n");

    float *input = malloc(sizeof(float) * fft_size);
    kiss_fft_cpx *fft_output = malloc(sizeof(kiss_fft_cpx) * (fft_size / 2 + 1));
    float *reconstructed = malloc(sizeof(float) * fft_size);
    float *power_spectrum = malloc(sizeof(float) * (fft_size / 2 + 1));

    if (!input || !fft_output || !reconstructed || !power_spectrum) {
        fprintf(stderr, "错误：内存分配失败\n");
        return 1;
    }

    generate_multi_tone_signal(input, fft_size, frequencies, num_freqs);
    apply_hamming_window(input, fft_size);

    printf("✓ 测试信号生成完成\n");
    print_real_array("输入信号（前 16 个样本）", input, 16);
    printf("\n");

    // 步骤 3：执行实数 FFT
    printf("步骤 3：执行实数正向 FFT\n");
    printf("─────────────────────────────────────────\n");

    kiss_fftr(fft_cfg, input, fft_output);
    printf("✓ 实数 FFT 计算完成\n\n");

    // 步骤 4：分析频谱
    printf("步骤 4：分析频谱\n");
    printf("─────────────────────────────────────────\n");

    print_fft_result(fft_output, fft_size);

    // 计算功率谱
    compute_power_spectrum(fft_output, power_spectrum, fft_size);

    // 找到峰值
    float peak_freq, peak_magnitude;
    analyze_spectrum(fft_output, fft_size, &peak_freq, &peak_magnitude);
    printf("峰值频率: %.1f Hz，幅值: %.6f\n\n", peak_freq, peak_magnitude);

    // 打印频谱图
    float *magnitude = malloc(sizeof(float) * (fft_size / 2 + 1));
    for (int i = 0; i < fft_size / 2 + 1; i++) {
        magnitude[i] = sqrtf(power_spectrum[i]);
    }

    // 归一化幅值
    float max_mag = 0;
    for (int i = 1; i < fft_size / 2 + 1; i++) {
        if (magnitude[i] > max_mag) max_mag = magnitude[i];
    }
    if (max_mag > 0) {
        for (int i = 0; i < fft_size / 2 + 1; i++) {
            magnitude[i] /= max_mag;
        }
    }

    print_spectrum_bars(magnitude, fft_size / 2 + 1, 0.01f);

    // 检测所有峰值频率
    printf("\n检测到的频率成分（幅值 > 峰值的 10%%）：\n");
    for (int i = 1; i < fft_size / 2 + 1; i++) {
        float freq = (float)i * SAMPLE_RATE / fft_size;
        float mag = magnitude[i];
        if (mag > 0.1f) {
            printf("  %6.0f Hz: 幅值 %.3f\n", freq, mag);
        }
    }
    printf("\n");

    // 步骤 5：执行逆向实数 FFT
    printf("步骤 5：执行逆向实数 FFT\n");
    printf("─────────────────────────────────────────\n");

    kiss_fftri(ifft_cfg, fft_output, reconstructed);
    printf("✓ 实数 IFFT 计算完成\n\n");

    // 归一化
    for (int i = 0; i < fft_size; i++) {
        reconstructed[i] /= fft_size;
    }

    print_real_array("重建信号（前 16 个样本）", reconstructed, 16);
    printf("\n");

    // 步骤 6：验证结果
    printf("步骤 6：验证结果\n");
    printf("─────────────────────────────────────────\n");

    double mse = 0;
    for (int i = 0; i < fft_size; i++) {
        double error = input[i] - reconstructed[i];
        mse += error * error;
    }
    mse /= fft_size;
    double rmse = sqrt(mse);

    printf("均方误差 (MSE): %.10f\n", mse);
    printf("均方根误差 (RMSE): %.10f\n", rmse);

    if (rmse < 0.0001) {
        printf("✓ 验证通过！重建信号与原始信号匹配。\n");
    } else {
        printf("✗ 警告：重建误差较大。\n");
    }
    printf("\n");

    // 步骤 7：性能比较
    printf("步骤 7：与复数 FFT 的比较\n");
    printf("─────────────────────────────────────────\n");

    printf("实数 FFT 的优势：\n");
    printf("  1. 内存使用：只需要 N/2+1 个复数输出，而不是 N 个\n");
    printf("  2. 计算速度：约为复数 FFT 的一半时间\n");
    printf("  3. 输入简化：直接使用实数数组，无需转换为复数\n");
    printf("\n");

    printf("何时使用实数 FFT：\n");
    printf("  ✓ 输入信号是实数（大多数实际应用）\n");
    printf("  ✓ 需要处理速度更快\n");
    printf("  ✓ 内存受限\n");
    printf("\n");

    printf("何时使用复数 FFT：\n");
    printf("  ✓ 输入信号是复数（如 IQ 信号、复基带信号）\n");
    printf("  ✓ 需要完整的频谱信息（包括负频率）\n");
    printf("\n");

    // 清理资源
    printf("清理资源...\n");
    free(input);
    free(fft_output);
    free(reconstructed);
    free(power_spectrum);
    free(magnitude);
    kiss_fftr_free(fft_cfg);
    kiss_fftr_free(ifft_cfg);
    printf("✓ 完成\n\n");

    printf("╔════════════════════════════════════════════════╗\n");
    printf("║   实数 FFT 示例程序成功完成！                  ║\n");
    printf("╚════════════════════════════════════════════════╝\n");

    return 0;
}
