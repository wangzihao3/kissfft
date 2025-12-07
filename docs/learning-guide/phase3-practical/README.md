# 阶段 3：实践应用

欢迎进入实践阶段！在这个阶段，我们将把理论和代码知识转化为实际的应用能力。

## 学习目标

完成本阶段后，您将能够：
- [ ] 编写使用 KISS FFT 的应用程序
- [ ] 实现实时频谱分析器
- [ ] 掌握性能优化技巧
- [ ] 集成 KISS FFT 到实际项目中
- [ ] 调试和优化 FFT 应用

## 实践项目概览

### 项目 1：基础 FFT 应用
- 简单的频谱分析器
- 音频信号处理
- 实时可视化

### 项目 2：音频效果器
- 均衡器实现
- 音频滤波
- 特效处理

### 项目 3：图像处理应用
- 图像频域滤波
- 图像压缩
- 特征提取

## 项目 1：音频频谱分析器

### 1.1 基础频谱分析器

```c
// spectrum_analyzer.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "kiss_fft.h"
#include <portaudio.h>

#define SAMPLE_RATE 44100
#define FFT_SIZE 1024
#define NUM_BINS 32

typedef struct {
    kiss_fft_cfg fft_cfg;
    kiss_fft_cpx *fft_in;
    kiss_fft_cpx *fft_out;
    float *window;
    float *magnitude;
    int buffer_pos;
} SpectrumAnalyzer;

// 初始化频谱分析器
SpectrumAnalyzer* spectrum_analyzer_init(int fft_size) {
    SpectrumAnalyzer *sa = malloc(sizeof(SpectrumAnalyzer));

    // 分配 FFT 配置
    sa->fft_cfg = kiss_fft_alloc(fft_size, 0, NULL, NULL);

    // 分配缓冲区
    sa->fft_in = malloc(sizeof(kiss_fft_cpx) * fft_size);
    sa->fft_out = malloc(sizeof(kiss_fft_cpx) * fft_size);
    sa->window = malloc(sizeof(float) * fft_size);
    sa->magnitude = malloc(sizeof(float) * fft_size / 2);

    // 生成汉明窗
    for (int i = 0; i < fft_size; i++) {
        sa->window[i] = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (fft_size - 1));
    }

    sa->buffer_pos = 0;

    return sa;
}

// 处理音频帧
void spectrum_analyzer_process(SpectrumAnalyzer *sa, const float *input, int frame_size) {
    for (int i = 0; i < frame_size; i++) {
        // 应用窗函数
        sa->fft_in[sa->buffer_pos].r = input[i] * sa->window[sa->buffer_pos];
        sa->fft_in[sa->buffer_pos].i = 0.0f;

        sa->buffer_pos++;

        // 缓冲区满时进行 FFT
        if (sa->buffer_pos >= FFT_SIZE) {
            sa->buffer_pos = 0;

            // 执行 FFT
            kiss_fft(sa->fft_cfg, sa->fft_in, sa->fft_out);

            // 计算幅值谱
            for (int k = 0; k < FFT_SIZE / 2; k++) {
                float real = sa->fft_out[k].r;
                float imag = sa->fft_out[k].i;
                sa->magnitude[k] = sqrtf(real * real + imag * imag) / FFT_SIZE;
            }
        }
    }
}

// 获取频带能量
void spectrum_analyzer_get_bands(SpectrumAnalyzer *sa, float *bands, int num_bands) {
    int samples_per_band = (FFT_SIZE / 2) / num_bands;

    for (int b = 0; b < num_bands; b++) {
        float energy = 0.0f;
        int start = b * samples_per_band;
        int end = start + samples_per_band;

        for (int k = start; k < end; k++) {
            energy += sa->magnitude[k] * sa->magnitude[k];
        }

        bands[b] = sqrtf(energy) / samples_per_band;
    }
}

// 清理资源
void spectrum_analyzer_free(SpectrumAnalyzer *sa) {
    if (sa) {
        kiss_fft_free(sa->fft_cfg);
        free(sa->fft_in);
        free(sa->fft_out);
        free(sa->window);
        free(sa->magnitude);
        free(sa);
    }
}

// 可视化频谱（文本模式）
void visualize_spectrum(float *bands, int num_bands) {
    const char *bars[] = {" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
    const int num_levels = sizeof(bars) / sizeof(bars[0]);

    // 找到最大值用于归一化
    float max_val = 0.001f;
    for (int i = 0; i < num_bands; i++) {
        if (bands[i] > max_val) max_val = bands[i];
    }

    // 打印频谱
    printf("\r");
    for (int i = 0; i < num_bands; i++) {
        float normalized = bands[i] / max_val;
        int level = (int)(normalized * (num_levels - 1));
        if (level < 0) level = 0;
        if (level >= num_levels) level = num_levels - 1;
        printf("%s", bars[level]);
    }
    fflush(stdout);
}
```

### 1.2 实时音频捕获

```c
// PortAudio 回调函数
static int audio_callback(const void *input_buffer, void *output_buffer,
                         unsigned long frames_per_buffer,
                         const PaStreamCallbackTimeInfo *time_info,
                         PaStreamCallbackFlags status_flags,
                         void *user_data) {
    SpectrumAnalyzer *sa = (SpectrumAnalyzer *)user_data;
    const float *in = (const float *)input_buffer;

    // 处理音频帧
    spectrum_analyzer_process(sa, in, frames_per_buffer);

    return paContinue;
}

// 主函数
int main() {
    // 初始化 PortAudio
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        printf("PortAudio error: %s\n", Pa_GetErrorText(err));
        return 1;
    }

    // 创建频谱分析器
    SpectrumAnalyzer *sa = spectrum_analyzer_init(FFT_SIZE);

    // 设置音频流参数
    PaStreamParameters input_params;
    input_params.device = Pa_GetDefaultInputDevice();
    input_params.channelCount = 1;  // 单声道
    input_params.sampleFormat = paFloat32;
    input_params.suggestedLatency = Pa_GetDeviceInfo(input_params.device)->defaultLowInputLatency;
    input_params.hostApiSpecificStreamInfo = NULL;

    // 打开音频流
    PaStream *stream;
    err = Pa_OpenStream(&stream, &input_params, NULL,
                       SAMPLE_RATE, 256, paClipOff,
                       audio_callback, sa);

    if (err != paNoError) {
        printf("PortAudio error: %s\n", Pa_GetErrorText(err));
        return 1;
    }

    // 开始音频流
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        printf("PortAudio error: %s\n", Pa_GetErrorText(err));
        return 1;
    }

    printf("Real-time Spectrum Analyzer\n");
    printf("Press Ctrl+C to exit\n");

    // 主循环 - 显示频谱
    float bands[NUM_BINS];
    while (1) {
        spectrum_analyzer_get_bands(sa, bands, NUM_BINS);
        visualize_spectrum(bands, NUM_BINS);
        Pa_Sleep(50);  // 20 FPS
    }

    // 清理
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    spectrum_analyzer_free(sa);

    return 0;
}
```

### 1.3 编译和运行

```makefile
# Makefile for spectrum analyzer
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2
LIBS = -lportaudio -lm -lpthread

# 包含 KISS FFT
KISSFFT_ROOT = ../..
CFLAGS += -I$(KISSFFT_ROOT)
KISSFFT_SRC = $(KISSFFT_ROOT)/kiss_fft.c

# 目标
TARGET = spectrum_analyzer
SOURCES = spectrum_analyzer.c $(KISSFFT_SRC)

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
```

## 项目 2：3 段均衡器

### 2.1 频域均衡器实现

```c
// equalizer.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kiss_fft.h"
#include "kiss_fftr.h"

#define SAMPLE_RATE 44100
#define BUFFER_SIZE 1024

typedef struct {
    // 滤波器参数
    float low_gain;      // 低频增益 (dB)
    float mid_gain;      // 中频增益 (dB)
    float high_gain;     // 高频增益 (dB)

    // FFT 相关
    kiss_fftr_cfg fft_cfg;   // 实数 FFT 配置
    kiss_fftr_cfg ifft_cfg;  // 实数 IFFT 配置
    float *fft_in;           // FFT 输入缓冲区
    kiss_fft_cpx *fft_out;   // FFT 输出缓冲区
    float *ifft_out;         // IFFT 输出缓冲区

    // 频域滤波器
    kiss_fft_cpx *filter;    // 频域滤波器响应

    // 缓冲区管理
    float *input_buffer;     // 输入环形缓冲区
    float *output_buffer;    // 输出缓冲区
    int buffer_pos;          // 缓冲区位置

} Equalizer;

// dB 转线性增益
static float db_to_linear(float db) {
    return powf(10.0f, db / 20.0f);
}

// 创建均衡器
Equalizer* equalizer_create(int buffer_size, float low_gain, float mid_gain, float high_gain) {
    Equalizer *eq = malloc(sizeof(Equalizer));

    eq->low_gain = low_gain;
    eq->mid_gain = mid_gain;
    eq->high_gain = high_gain;

    // 创建 FFT 配置
    eq->fft_cfg = kiss_fftr_alloc(buffer_size, 0, NULL, NULL);
    eq->ifft_cfg = kiss_fftr_alloc(buffer_size, 1, NULL, NULL);

    // 分配缓冲区
    eq->fft_in = malloc(sizeof(float) * buffer_size);
    eq->fft_out = malloc(sizeof(kiss_fft_cpx) * (buffer_size/2 + 1));
    eq->ifft_out = malloc(sizeof(float) * buffer_size);
    eq->filter = malloc(sizeof(kiss_fft_cpx) * (buffer_size/2 + 1));

    eq->input_buffer = malloc(sizeof(float) * buffer_size);
    eq->output_buffer = malloc(sizeof(float) * buffer_size);
    eq->buffer_pos = 0;

    // 初始化缓冲区
    memset(eq->input_buffer, 0, sizeof(float) * buffer_size);
    memset(eq->output_buffer, 0, sizeof(float) * buffer_size);

    // 创建频域滤波器
    equalizer_update_filter(eq, buffer_size);

    return eq;
}

// 更新滤波器响应
void equalizer_update_filter(Equalizer *eq, int buffer_size) {
    int num_bins = buffer_size / 2 + 1;

    for (int k = 0; k < num_bins; k++) {
        float freq = (float)k * SAMPLE_RATE / buffer_size;

        // 计算每个频段的增益
        float gain = 1.0f;

        // 低频 (20-250 Hz)
        if (freq < 250.0f) {
            float weight = 1.0f - (freq - 20.0f) / 230.0f;
            gain = 1.0f + (db_to_linear(eq->low_gain) - 1.0f) * weight;
        }
        // 中频 (250-4000 Hz)
        else if (freq < 4000.0f) {
            float weight = 1.0f - fabsf(freq - 1000.0f) / 3000.0f;
            gain = 1.0f + (db_to_linear(eq->mid_gain) - 1.0f) * weight;
        }
        // 高频 (4000-20000 Hz)
        else if (freq < 20000.0f) {
            float weight = (20000.0f - freq) / 16000.0f;
            gain = 1.0f + (db_to_linear(eq->high_gain) - 1.0f) * weight;
        }

        eq->filter[k].r = gain;
        eq->filter[k].i = 0.0f;
    }
}

// 处理音频样本
int equalizer_process(Equalizer *eq, float *input, float *output, int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        // 存储输入样本
        eq->input_buffer[eq->buffer_pos] = input[i];

        // 输出处理过的样本（延迟）
        output[i] = eq->output_buffer[eq->buffer_pos];

        eq->buffer_pos++;

        // 缓冲区满时进行处理
        if (eq->buffer_pos >= BUFFER_SIZE) {
            eq->buffer_pos = 0;

            // 1. 复制到 FFT 输入（应用窗函数）
            for (int j = 0; j < BUFFER_SIZE; j++) {
                // 汉明窗
                float window = 0.54f - 0.46f * cosf(2.0f * M_PI * j / (BUFFER_SIZE - 1));
                eq->fft_in[j] = eq->input_buffer[j] * window;
            }

            // 2. 执行 FFT
            kiss_fftr(eq->fft_cfg, eq->fft_in, eq->fft_out);

            // 3. 应用频域滤波器
            for (int k = 0; k < BUFFER_SIZE/2 + 1; k++) {
                eq->fft_out[k].r *= eq->filter[k].r;
                eq->fft_out[k].i *= eq->filter[k].r;
            }

            // 4. 执行 IFFT
            kiss_fftri(eq->ifft_cfg, eq->fft_out, eq->ifft_out);

            // 5. 归一化并存储到输出缓冲区
            for (int j = 0; j < BUFFER_SIZE; j++) {
                eq->output_buffer[j] = eq->ifft_out[j] / BUFFER_SIZE;
            }
        }
    }

    return num_samples;
}

// 设置均衡器增益
void equalizer_set_gains(Equalizer *eq, float low, float mid, float high) {
    eq->low_gain = low;
    eq->mid_gain = mid;
    eq->high_gain = high;
    equalizer_update_filter(eq, BUFFER_SIZE);
}

// 清理均衡器
void equalizer_destroy(Equalizer *eq) {
    if (eq) {
        kiss_fftr_free(eq->fft_cfg);
        kiss_fftr_free(eq->ifft_cfg);
        free(eq->fft_in);
        free(eq->fft_out);
        free(eq->ifft_out);
        free(eq->filter);
        free(eq->input_buffer);
        free(eq->output_buffer);
        free(eq);
    }
}
```

### 2.2 交互式均衡器测试

```c
// interactive_equalizer.c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>

// 非阻塞键盘输入
int kbhit(void) {
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if(ch != EOF) {
        ungetc(ch, stdin);
        return 1;
    }

    return 0;
}

// 主测试函数
int main() {
    // 创建均衡器
    Equalizer *eq = equalizer_create(BUFFER_SIZE, 0.0f, 0.0f, 0.0f);

    // 测试信号
    float test_input[BUFFER_SIZE];
    float test_output[BUFFER_SIZE];

    printf("=== Interactive 3-Band Equalizer ===\n");
    printf("Controls:\n");
    printf("  q/a: Low frequency gain (-12 dB to +12 dB)\n");
    printf("  w/s: Mid frequency gain (-12 dB to +12 dB)\n");
    printf("  e/d: High frequency gain (-12 dB to +12 dB)\n");
    printf("  r  : Reset all gains to 0 dB\n");
    printf("  x  : Exit\n\n");

    float low_gain = 0.0f, mid_gain = 0.0f, high_gain = 0.0f;

    while (1) {
        // 显示当前增益设置
        printf("\rLow: %+5.1f dB  Mid: %+5.1f dB  High: %+5.1f dB",
               low_gain, mid_gain, high_gain);
        fflush(stdout);

        // 检查键盘输入
        if (kbhit()) {
            char c = getchar();

            switch (c) {
                case 'q':
                    low_gain = fminf(12.0f, low_gain + 1.0f);
                    equalizer_set_gains(eq, low_gain, mid_gain, high_gain);
                    break;
                case 'a':
                    low_gain = fmaxf(-12.0f, low_gain - 1.0f);
                    equalizer_set_gains(eq, low_gain, mid_gain, high_gain);
                    break;
                case 'w':
                    mid_gain = fminf(12.0f, mid_gain + 1.0f);
                    equalizer_set_gains(eq, low_gain, mid_gain, high_gain);
                    break;
                case 's':
                    mid_gain = fmaxf(-12.0f, mid_gain - 1.0f);
                    equalizer_set_gains(eq, low_gain, mid_gain, high_gain);
                    break;
                case 'e':
                    high_gain = fminf(12.0f, high_gain + 1.0f);
                    equalizer_set_gains(eq, low_gain, mid_gain, high_gain);
                    break;
                case 'd':
                    high_gain = fmaxf(-12.0f, high_gain - 1.0f);
                    equalizer_set_gains(eq, low_gain, mid_gain, high_gain);
                    break;
                case 'r':
                    low_gain = mid_gain = high_gain = 0.0f;
                    equalizer_set_gains(eq, low_gain, mid_gain, high_gain);
                    break;
                case 'x':
                    printf("\nExiting...\n");
                    equalizer_destroy(eq);
                    return 0;
            }
        }

        // 生成测试信号（混合频率）
        for (int i = 0; i < 256; i++) {
            float t = (float)i / SAMPLE_RATE;
            test_input[i] = sinf(2 * M_PI * 100 * t) +    // 低频
                           0.5f * sinf(2 * M_PI * 1000 * t) +  // 中频
                           0.3f * sinf(2 * M_PI * 8000 * t);   // 高频
        }

        // 处理信号
        equalizer_process(eq, test_input, test_output, 256);

        usleep(50000);  // 50ms
    }

    return 0;
}
```

## 性能优化实践

### 1. 使用 OpenMP 并行化

```c
// 并行化版本的性能测试
#include <omp.h>

void parallel_fft_test() {
    const int N = 4096;
    const int iterations = 10000;
    kiss_fft_cfg cfg = kiss_fft_alloc(N, 0, NULL, NULL);

    kiss_fft_cpx *input = malloc(sizeof(kiss_fft_cpx) * N);
    kiss_fft_cpx *output = malloc(sizeof(kiss_fft_cpx) * N);

    // 初始化随机数据
    srand(42);
    for (int i = 0; i < N; i++) {
        input[i].r = (float)rand() / RAND_MAX;
        input[i].i = (float)rand() / RAND_MAX;
    }

    printf("=== Parallel FFT Performance Test ===\n");

    // 测试不同线程数
    for (int num_threads = 1; num_threads <= 8; num_threads *= 2) {
        omp_set_num_threads(num_threads);

        double start_time = omp_get_wtime();

        #pragma omp parallel for
        for (int i = 0; i < iterations; i++) {
            kiss_fft(cfg, input, output);
        }

        double end_time = omp_get_wtime();
        double avg_time = (end_time - start_time) * 1000 / iterations;

        printf("Threads: %2d, Average time: %8.3f ms, Speedup: %5.2fx\n",
               num_threads, avg_time, avg_time / avg_time);
    }

    free(input);
    free(output);
    kiss_fft_free(cfg);
}
```

### 2. SIMD 优化测试

```c
// SIMD 版本的点积计算
#include <immintrin.h>

void simd_vector_dot_product(float *a, float *b, float *result, int n) {
    __m256 sum = _mm256_setzero_ps();

    // 处理8个元素为一组
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        __m256 va = _mm256_load_ps(&a[i]);
        __m256 vb = _mm256_load_ps(&b[i]);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
    }

    // 水平求和
    float temp[8];
    _mm256_store_ps(temp, sum);

    *result = 0.0f;
    for (int j = 0; j < 8; j++) {
        *result += temp[j];
    }

    // 处理剩余元素
    for (; i < n; i++) {
        *result += a[i] * b[i];
    }
}

// 性能对比测试
void compare_dot_product_performance() {
    const int N = 1024 * 1024;
    float *a = aligned_alloc(32, N * sizeof(float));
    float *b = aligned_alloc(32, N * sizeof(float));

    // 初始化数据
    for (int i = 0; i < N; i++) {
        a[i] = (float)i / N;
        b[i] = 1.0f - (float)i / N;
    }

    float result;
    int iterations = 1000;

    // 标准版本
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        result = 0.0f;
        for (int j = 0; j < N; j++) {
            result += a[j] * b[j];
        }
    }
    clock_t end = clock();
    double standard_time = (double)(end - start) / CLOCKS_PER_SEC;

    // SIMD 版本
    start = clock();
    for (int i = 0; i < iterations; i++) {
        simd_vector_dot_product(a, b, &result, N);
    }
    end = clock();
    double simd_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Standard dot product: %6.3f ms\n", standard_time * 1000 / iterations);
    printf("SIMD dot product:     %6.3f ms\n", simd_time * 1000 / iterations);
    printf("Speedup:              %.2fx\n", standard_time / simd_time);

    free(a);
    free(b);
}
```

## 本周实践任务

### 第 6 周：基础应用开发

**周一/周二**
- [ ] 编译并运行频谱分析器
- [ ] 理解实时音频处理流程
- [ ] 修改窗函数类型，观察效果

**周三/周四**
- [ ] 实现 3 段均衡器
- [ ] 测试不同增益设置的效果
- [ ] 添加可视化的频谱显示

**周五**
- [ ] 集成两个模块
- [ ] 性能测试和优化
- [ ] 编写使用文档

### 第 7 周：性能优化和集成

**周一/周二**
- [ ] 使用性能分析工具找出瓶颈
- [ ] 实现 SIMD 优化（如果可用）
- [ ] 测试多核并行化

**周三/周四**
- [ ] 集成到实际项目中
- [ ] 处理边界情况
- [ ] 优化内存使用

**周五**
- [ ] 完整测试
- [ ] 代码审查
- [ ] 准备演示

## 项目评估

### 功能完整性
- [ ] 实时音频处理
- [ ] 可视化显示
- [ ] 交互控制
- [ ] 错误处理

### 性能指标
- [ ] 延迟 < 20ms
- [ ] CPU 使用率 < 50%
- [ ] 内存使用合理
- [ ] 稳定运行

### 代码质量
- [ ] 清晰的模块划分
- [ ] 充分的注释
- [ ] 错误处理
- [ ] 可扩展性

## 下一阶段预告

完成本阶段后，您将具备：
- 实际开发 FFT 应用的能力
- 性能优化和调试技能
- 项目集成和部署经验

准备好进入[阶段 4：高级主题](../phase4-advanced/)，探索更深入的主题！

---

记住：实践是学习的最佳方式，不断尝试和改进才能达到精通。