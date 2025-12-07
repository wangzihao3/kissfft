# 阶段 4：高级主题

欢迎进入高级主题阶段！在这个阶段，我们将探索 KISS FFT 的进阶特性和优化技术。

## 学习目标

完成本阶段后，您将能够：
- [ ] 掌握多维 FFT 的实现和应用
- [ ] 理解 SIMD 优化技术
- [ ] 使用 OpenMP 进行并行计算
- [ ] 实现 FFT 工具链集成
- [ ] 扩展和定制 KISS FFT

## 高级主题概览

### 1. 多维 FFT
- 2D/3D FFT 实现
- 图像处理应用
- 数据布局优化

### 2. SIMD 优化
- SSE/AVX 指令集
- 向量化运算
- 性能分析

### 3. 并行计算
- OpenMP 并行化
- 多线程策略
- 负载均衡

### 4. 工具和扩展
- 快速卷积
- 实用工具
- 集成方案

## 多维 FFT

### 1.1 理解多维 FFT

多维 FFT 是一维 FFT 的自然扩展，用于处理多维信号（如图像）。

**数学定义：**
```
2D FFT:  X[k,l] = Σx Σy x[x,y] * e^(-j2π(kx/Nx + ly/Ny))
```

**实现策略：**
1. 先对每一行进行 1D FFT
2. 再对每一列进行 1D FFT
3. 或者利用分离性质进行优化

### 1.2 使用 kiss_fftnd

```c
// multidim_fft.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "kiss_fftnd.h"

// 创建 2D 图像
void create_test_image(float *image, int width, int height, int pattern) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;

            switch (pattern) {
                case 0: // 水平条纹
                    image[idx] = sinf(2 * M_PI * y / height);
                    break;
                case 1: // 垂直条纹
                    image[idx] = sinf(2 * M_PI * x / width);
                    break;
                case 2: // 棋盘格
                    image[idx] = ((x/16 + y/16) % 2) * 1.0f - 0.5f;
                    break;
                case 3: // 圆形图案
                    float cx = width / 2.0f;
                    float cy = height / 2.0f;
                    float r = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
                    image[idx] = sinf(2 * M_PI * r / 32);
                    break;
            }
        }
    }
}

// 2D FFT 包装函数
void fft2d(float *input, kiss_fft_cpx *output, int width, int height) {
    // 创建 2D FFT 配置
    int dims[2] = {height, width};
    kiss_fftnd_cfg cfg = kiss_fftnd_alloc(dims, 2, 0, NULL, NULL);

    // 转换为复数格式
    kiss_fft_cpx *complex_input = malloc(sizeof(kiss_fft_cpx) * width * height);
    for (int i = 0; i < width * height; i++) {
        complex_input[i].r = input[i];
        complex_input[i].i = 0.0f;
    }

    // 执行 2D FFT
    kiss_fftnd(cfg, complex_input, output);

    free(complex_input);
    kiss_fftnd_free(cfg);
}

// 2D IFFT 包装函数
void ifft2d(kiss_fft_cpx *input, float *output, int width, int height) {
    // 创建 2D IFFT 配置
    int dims[2] = {height, width};
    kiss_fftnd_cfg cfg = kiss_fftnd_alloc(dims, 2, 1, NULL, NULL);

    kiss_fft_cpx *complex_output = malloc(sizeof(kiss_fft_cpx) * width * height);

    // 执行 2D IFFT
    kiss_fftnd(cfg, input, complex_output);

    // 提取实部并归一化
    for (int i = 0; i < width * height; i++) {
        output[i] = complex_output[i].r / (width * height);
    }

    free(complex_output);
    kiss_fftnd_free(cfg);
}

// 保存图像为 PGM 格式
void save_image_pgm(const char *filename, float *image, int width, int height) {
    FILE *file = fopen(filename, "w");
    if (!file) return;

    fprintf(file, "P2\n%d %d\n255\n", width, height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            float pixel = (image[idx] + 1.0f) * 127.5f; // 映射到 0-255
            pixel = fmaxf(0.0f, fminf(255.0f, pixel));
            fprintf(file, "%d ", (int)pixel);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

// 频域滤波
void frequency_domain_filter(kiss_fft_cpx *spectrum, int width, int height,
                           int filter_type, float cutoff) {
    int center_x = width / 2;
    int center_y = height / 2;
    float max_radius = sqrtf(center_x * center_x + center_y * center_y);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;

            // 计算到中心的距离
            float dx = (x >= center_x) ? x - center_x : center_x - x;
            float dy = (y >= center_y) ? y - center_y : center_y - y;
            float radius = sqrtf(dx * dx + dy * dy);

            float filter_gain = 1.0f;

            switch (filter_type) {
                case 0: // 低通滤波器
                    filter_gain = radius <= cutoff * max_radius ? 1.0f : 0.0f;
                    break;
                case 1: // 高通滤波器
                    filter_gain = radius <= cutoff * max_radius ? 0.0f : 1.0f;
                    break;
                case 2: // 带通滤波器
                    filter_gain = (radius >= cutoff * 0.3f * max_radius &&
                                 radius <= cutoff * 0.7f * max_radius) ? 1.0f : 0.0f;
                    break;
                case 3: // 高斯低通
                    filter_gain = expf(-0.5f * (radius / (cutoff * max_radius)) *
                                    (radius / (cutoff * max_radius)));
                    break;
            }

            spectrum[idx].r *= filter_gain;
            spectrum[idx].i *= filter_gain;
        }
    }
}

int main() {
    const int width = 256;
    const int height = 256;

    printf("=== 2D FFT Demo ===\n");

    // 分配内存
    float *image = malloc(sizeof(float) * width * height);
    float *filtered_image = malloc(sizeof(float) * width * height);
    kiss_fft_cpx *spectrum = malloc(sizeof(kiss_fft_cpx) * width * height);

    // 创建测试图像
    printf("Creating test image (chess pattern)...\n");
    create_test_image(image, width, height, 2); // 棋盘格
    save_image_pgm("input.pgm", image, width, height);

    // 执行 2D FFT
    printf("Computing 2D FFT...\n");
    fft2d(image, spectrum, width, height);

    // 应用频域滤波
    printf("Applying low-pass filter...\n");
    frequency_domain_filter(spectrum, width, height, 0, 0.1); // 低通

    // 执行 2D IFFT
    printf("Computing 2D IFFT...\n");
    ifft2d(spectrum, filtered_image, width, height);
    save_image_pgm("filtered.pgm", filtered_image, width, height);

    // 清理
    free(image);
    free(filtered_image);
    free(spectrum);

    printf("Done! Check input.pgm and filtered.pgm\n");
    return 0;
}
```

### 1.3 实数多维 FFT

对于实数图像，可以使用 `kiss_fftndr` 进行优化：

```c
// real_multidim_fft.c
#include "kiss_fftndr.h"

void real_fft2d(float *input, kiss_fft_cpx *output, int width, int height) {
    int dims[2] = {height, width};
    kiss_fftndr_cfg cfg = kiss_fftndr_alloc(dims, 2, 0, NULL, NULL);

    kiss_fftndr(cfg, input, output);

    kiss_fftndr_free(cfg);
}

void real_ifft2d(kiss_fft_cpx *input, float *output, int width, int height) {
    int dims[2] = {height, width};
    kiss_fftndr_cfg cfg = kiss_fftndr_alloc(dims, 2, 1, NULL, NULL);

    kiss_fftndri(cfg, input, output);

    kiss_fftndr_free(cfg);
}
```

## SIMD 优化

### 2.1 理解 SIMD

SIMD (Single Instruction, Multiple Data) 允许用一条指令处理多个数据：

```c
// simd_fft.c
#include <immintrin.h>
#include "kiss_fft.h"

// SIMD 优化的复数乘法
inline void simd_complex_multiply(kiss_fft_cpx *result,
                                const kiss_fft_cpx *a,
                                const kiss_fft_cpx *b) {
    __m128 va = _mm_set_ps(a[1].i, a[1].r, a[0].i, a[0].r);
    __m128 vb = _mm_set_ps(b[1].i, b[1].r, b[0].r, b[0].i);
    __m128 vb_swap = _mm_set_ps(b[1].r, b[1].i, b[0].i, b[0].r);

    __m128 mul1 = _mm_mul_ps(va, vb);
    __m128 mul2 = _mm_mul_ps(va, vb_swap);

    // 结果: (a.r*b.r - a.i*b.i, a.r*b.i + a.i*b.r)
    __m128 result_lo = _mm_sub_ps(mul1, mul2);
    __m128 result_hi = _mm_add_ps(mul1, _mm2);

    float tmp[4];
    _mm_store_ps(tmp, result_lo);
    result[0].r = tmp[0];
    result[0].i = tmp[1];

    _mm_store_ps(tmp, result_hi);
    result[1].r = tmp[0];
    result[1].i = tmp[1];
}

// SIMD 优化的蝶形运算
void simd_butterfly2(kiss_fft_cpx *Fout, const kiss_fft_cpx *tw,
                    int m, int stride) {
    // 处理两个蝶形运算
    for (int i = 0; i < m; i += 2) {
        // 加载输入数据
        __m128 vin = _mm_loadu_ps(&Fout[i]);  // 加载两个复数
        __m128 vtw = _mm_loadu_ps(&tw[i]);    // 加载两个旋转因子

        // 实际的 SIMD 蝶形运算实现
        // 这里简化了，完整实现需要更复杂的 SIMD 操作

        // 保存结果
        _mm_storeu_ps(&Fout[i], vin);
    }
}
```

### 2.2 性能对比

```c
// performance_comparison.c
#include <time.h>

#define NUM_ITERATIONS 10000
#define FFT_SIZE 1024

// 标准 FFT 函数（调用 kiss_fft）
void standard_fft(kiss_fft_cfg cfg, kiss_fft_cpx *input,
                  kiss_fft_cpx *output) {
    kiss_fft(cfg, input, output);
}

// SIMD 优化版本（假设已实现）
void simd_fft(kiss_fft_cfg cfg, kiss_fft_cpx *input,
              kiss_fft_cpx *output) {
    // 这里是 SIMD 优化版本
    // 实际实现会更复杂
    standard_fft(cfg, input, output);  // 暂时使用标准版本
}

void benchmark_fft() {
    kiss_fft_cfg cfg = kiss_fft_alloc(FFT_SIZE, 0, NULL, NULL);

    kiss_fft_cpx *input = malloc(sizeof(kiss_fft_cpx) * FFT_SIZE);
    kiss_fft_cpx *output = malloc(sizeof(kiss_fft_cpx) * FFT_SIZE);

    // 初始化测试数据
    for (int i = 0; i < FFT_SIZE; i++) {
        input[i].r = (float)rand() / RAND_MAX;
        input[i].i = (float)rand() / RAND_MAX;
    }

    printf("=== FFT Performance Comparison ===\n");

    // 标准版本测试
    clock_t start = clock();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        standard_fft(cfg, input, output);
    }
    clock_t end = clock();
    double standard_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Standard FFT:   %8.3f seconds (%8.3f ms per FFT)\n",
           standard_time, standard_time * 1000 / NUM_ITERATIONS);

    // SIMD 版本测试
    start = clock();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        simd_fft(cfg, input, output);
    }
    end = clock();
    double simd_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("SIMD FFT:       %8.3f seconds (%8.3f ms per FFT)\n",
           simd_time, simd_time * 1000 / NUM_ITERATIONS);

    printf("Speedup:        %.2fx\n", standard_time / simd_time);

    free(input);
    free(output);
    kiss_fft_free(cfg);
}
```

## OpenMP 并行化

### 3.1 FFT 的并行策略

```c
// openmp_fft.c
#include <omp.h>

// 并行处理多个 FFT
void parallel_multiple_ffts() {
    const int num_ffts = 100;
    const int fft_size = 1024;

    #pragma omp parallel
    {
        // 每个线程有自己的配置
        kiss_fft_cfg cfg = kiss_fft_alloc(fft_size, 0, NULL, NULL);

        kiss_fft_cpx *input = malloc(sizeof(kiss_fft_cpx) * fft_size);
        kiss_fft_cpx *output = malloc(sizeof(kiss_fft_cpx) * fft_size);

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < num_ffts; i++) {
            // 初始化数据
            for (int j = 0; j < fft_size; j++) {
                input[j].r = sinf(2 * M_PI * j / fft_size * (i + 1));
                input[j].i = 0;
            }

            // 执行 FFT
            kiss_fft(cfg, input, output);
        }

        free(input);
        free(output);
        kiss_fft_free(cfg);
    }
}

// 大 FFT 的并行化（更复杂，需要特殊算法）
void parallel_large_fft(kiss_fft_cfg cfg, kiss_fft_cpx *input,
                      kiss_fft_cpx *output, int N) {
    // 这里展示概念，实际实现需要重新设计算法

    // 将大 FFT 分解为可以并行执行的小 FFT
    if (N > 4096) {
        int sub_size = N / 4;

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                // 处理第一个四分之一
                // 需要自定义的并行 FFT 实现
            }
            #pragma omp section
            {
                // 处理第二个四分之一
            }
            #pragma omp section
            {
                // 处理第三个四分之一
            }
            #pragma omp section
            {
                // 处理第四个四分之一
            }
        }

        // 合并结果
    } else {
        // 小 FFT 直接执行
        kiss_fft(cfg, input, output);
    }
}
```

### 3.2 性能分析

```c
// openmp_performance.c
void analyze_omp_performance() {
    const int sizes[] = {256, 512, 1024, 2048, 4096};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("=== OpenMP Performance Analysis ===\n");
    printf("Threads |  Size  |   Time   |  Speedup\n");
    printf("--------|--------|----------|---------\n");

    for (int nthreads = 1; nthreads <= 8; nthreads *= 2) {
        omp_set_num_threads(nthreads);

        for (int s = 0; s < num_sizes; s++) {
            int N = sizes[s];
            kiss_fft_cfg cfg = kiss_fft_alloc(N, 0, NULL, NULL);

            // 测试时间
            double start_time = omp_get_wtime();

            #pragma omp parallel for
            for (int i = 0; i < 1000; i++) {
                kiss_fft_cpx *input = malloc(sizeof(kiss_fft_cpx) * N);
                kiss_fft_cpx *output = malloc(sizeof(kiss_fft_cpx) * N);

                kiss_fft(cfg, input, output);

                free(input);
                free(output);
            }

            double end_time = omp_get_wtime();
            double elapsed = end_time - start_time;

            // 打印结果
            if (nthreads == 1) {
                printf("%8d | %6d | %8.3f | %7.2fx\n",
                       nthreads, N, elapsed * 1000, 1.0);
            }

            kiss_fft_free(cfg);
        }
    }
}
```

## 快速卷积

### 4.1 使用 kiss_fastfir

```c
// fast_convolution.c
#include "kiss_fft.h"
#include "kiss_fftr.h"

// 实现快速卷积（使用 FFT）
void fast_convolution(const float *x, int nx,
                    const float *h, int nh,
                    float *y) {
    // 计算输出长度
    int ny = nx + nh - 1;

    // 找到合适的 FFT 长度（2 的幂）
    int n = 1;
    while (n < ny) n *= 2;

    // 创建实数 FFT 配置
    kiss_fftr_cfg fft_cfg = kiss_fftr_alloc(n, 0, NULL, NULL);
    kiss_fftr_cfg ifft_cfg = kiss_fftr_alloc(n, 1, NULL, NULL);

    // 分配缓冲区
    float *x_padded = calloc(n, sizeof(float));
    float *h_padded = calloc(n, sizeof(float));
    kiss_fft_cpx *X = malloc(sizeof(kiss_fft_cpx) * (n/2 + 1));
    kiss_fft_cpx *H = malloc(sizeof(kiss_fft_cpx) * (n/2 + 1));
    kiss_fft_cpx *Y = malloc(sizeof(kiss_fft_cpx) * (n/2 + 1));
    float *y_padded = malloc(sizeof(float) * n);

    // 复制输入并补零
    memcpy(x_padded, x, nx * sizeof(float));
    memcpy(h_padded, h, nh * sizeof(float));

    // FFT
    kiss_fftr(fft_cfg, x_padded, X);
    kiss_fftr(fft_cfg, h_padded, H);

    // 频域相乘
    for (int k = 0; k < n/2 + 1; k++) {
        // 复数乘法
        float real = X[k].r * H[k].r - X[k].i * H[k].i;
        float imag = X[k].r * H[k].i + X[k].i * H[k].r;
        Y[k].r = real;
        Y[k].i = imag;
    }

    // IFFT
    kiss_fftri(ifft_cfg, Y, y_padded);

    // 归一化并复制结果
    for (int i = 0; i < ny; i++) {
        y[i] = y_padded[i] / n;
    }

    // 清理
    free(x_padded);
    free(h_padded);
    free(X);
    free(H);
    free(Y);
    free(y_padded);
    kiss_fftr_free(fft_cfg);
    kiss_fftr_free(ifft_cfg);
}

// 测试快速卷积
void test_fast_convolution() {
    // 测试信号：两个方波的卷积应该是三角波
    const int N = 32;
    float x[N] = {0};
    float h[N] = {0};

    // 创建方波
    for (int i = 0; i < N/4; i++) {
        x[i] = 1.0f;
        h[i] = 1.0f;
    }

    float y[N*2 - 1];
    fast_convolution(x, N, h, N, y);

    // 打印结果
    printf("Fast Convolution Result:\n");
    for (int i = 0; i < N*2 - 1; i++) {
        printf("%2d: %6.3f\n", i, y[i]);
    }
}
```

### 4.2 实时卷积（重叠保留法）

```c
// realtime_convolution.c
typedef struct {
    float *filter;        // 滤波器脉冲响应
    int filter_len;       // 滤波器长度

    float *buffer;        // 输入缓冲区
    int buffer_pos;       // 缓冲区位置
    int block_size;       // 块大小

    kiss_fftr_cfg fft_cfg;
    kiss_fftr_cfg ifft_cfg;
    kiss_fft_cpx *filter_fft;
    float *fft_buffer;
    kiss_fft_cpx *fft_result;
} RealtimeConvolver;

RealtimeConvolver* convolver_create(const float *filter, int filter_len,
                                   int block_size) {
    RealtimeConvolver *c = malloc(sizeof(RealtimeConvolver));

    c->filter_len = filter_len;
    c->block_size = block_size;
    c->buffer_pos = 0;

    // FFT 长度
    int fft_len = block_size + filter_len - 1;
    int n = 1;
    while (n < fft_len) n *= 2;
    int fft_size = n;

    // 分配内存
    c->filter = malloc(sizeof(float) * filter_len);
    memcpy(c->filter, filter, filter_len * sizeof(float));

    c->buffer = calloc(fft_size, sizeof(float));
    c->fft_buffer = malloc(sizeof(float) * fft_size);
    c->fft_result = malloc(sizeof(kiss_fft_cpx) * (fft_size/2 + 1));

    // 创建 FFT 配置
    c->fft_cfg = kiss_fftr_alloc(fft_size, 0, NULL, NULL);
    c->ifft_cfg = kiss_fftr_alloc(fft_size, 1, NULL, NULL);

    // 预计算滤波器的 FFT
    float *filter_padded = calloc(fft_size, sizeof(float));
    memcpy(filter_padded, filter, filter_len * sizeof(float));
    c->filter_fft = malloc(sizeof(kiss_fft_cpx) * (fft_size/2 + 1));
    kiss_fftr(c->fft_cfg, filter_padded, c->filter_fft);
    free(filter_padded);

    return c;
}

void convolver_process(RealtimeConvolver *c, const float *input,
                      float *output, int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        c->buffer[c->buffer_pos] = input[i];
        c->buffer_pos++;

        // 当缓冲区满时处理
        if (c->buffer_pos >= c->block_size) {
            // 复制到 FFT 缓冲区
            memcpy(c->fft_buffer, c->buffer,
                   c->block_size * sizeof(float));

            // FFT
            kiss_fftr(c->fft_cfg, c->fft_buffer, c->fft_result);

            // 频域相乘
            for (int k = 0; k < c->fft_size/2 + 1; k++) {
                float real = c->fft_result[k].r * c->filter_fft[k].r -
                           c->fft_result[k].i * c->filter_fft[k].i;
                float imag = c->fft_result[k].r * c->filter_fft[k].i +
                           c->fft_result[k].i * c->filter_fft[k].r;
                c->fft_result[k].r = real;
                c->fft_result[k].i = imag;
            }

            // IFFT
            kiss_fftri(c->ifft_cfg, c->fft_result, c->fft_buffer);

            // 归一化
            for (int j = 0; j < c->block_size; j++) {
                c->fft_buffer[j] /= c->fft_size;
            }

            // 保存重叠部分
            for (int j = 0; j < c->filter_len - 1; j++) {
                c->buffer[j] += c->fft_buffer[c->block_size + j];
            }

            // 输出有效部分
            for (int j = 0; j < c->block_size; j++) {
                output[i - num_samples + j] = c->fft_buffer[j];
            }

            // 移动缓冲区
            memmove(c->buffer, c->buffer + c->block_size,
                   (c->filter_len - 1) * sizeof(float));
            c->buffer_pos = c->filter_len - 1;
        }
    }
}
```

## 工具集成

### 5.1 命令行工具集成

```c
// fft_toolchain.c
#include <getopt.h>

typedef struct {
    int fft_size;
    int inverse;
    int real;
    int verbose;
    char *input_file;
    char *output_file;
} FFTConfig;

void print_usage(const char *program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -s, --size <N>       FFT size (default: 1024)\n");
    printf("  -i, --inverse        Inverse FFT\n");
    printf("  -r, --real          Real FFT\n");
    printf("  -v, --verbose       Verbose output\n");
    printf("  -o, --output <file> Output file\n");
    printf("  -h, --help          Show this help\n");
}

int main(int argc, char *argv[]) {
    FFTConfig config = {
        .fft_size = 1024,
        .inverse = 0,
        .real = 0,
        .verbose = 0,
        .input_file = NULL,
        .output_file = NULL
    };

    // 解析命令行参数
    static struct option long_options[] = {
        {"size", required_argument, 0, 's'},
        {"inverse", no_argument, 0, 'i'},
        {"real", no_argument, 0, 'r'},
        {"verbose", no_argument, 0, 'v'},
        {"output", required_argument, 0, 'o'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "s:irvo:h", long_options, NULL)) != -1) {
        switch (c) {
            case 's':
                config.fft_size = atoi(optarg);
                break;
            case 'i':
                config.inverse = 1;
                break;
            case 'r':
                config.real = 1;
                break;
            case 'v':
                config.verbose = 1;
                break;
            case 'o':
                config.output_file = optarg;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    // 处理输入文件
    if (optind < argc) {
        config.input_file = argv[optind];
    }

    if (config.verbose) {
        printf("Configuration:\n");
        printf("  FFT size: %d\n", config.fft_size);
        printf("  Inverse: %s\n", config.inverse ? "yes" : "no");
        printf("  Real: %s\n", config.real ? "yes" : "no");
        if (config.input_file) {
            printf("  Input: %s\n", config.input_file);
        }
        if (config.output_file) {
            printf("  Output: %s\n", config.output_file);
        }
    }

    // 这里可以添加实际的 FFT 处理逻辑

    return 0;
}
```

## 本周学习任务

### 第 8 周：高级特性

**周一/周二**
- [ ] 实现和测试 2D FFT
- [ ] 应用图像滤波
- [ ] 比较实数和复数版本

**周三/周四**
- [ ] 实验 SIMD 优化
- [ ] 性能分析和对比
- [ ] 理解对齐要求

**周五**
- [ ] 并行化实验
- [ ] 分析可扩展性
- [ ] 总结优化技巧

### 第 9 周：集成和扩展

**周一/周二**
- [ ] 实现快速卷积
- [ ] 实时处理测试
- [ ] 优化延迟

**周三/周四**
- [ ] 集成所有工具
- [ ] 创建完整的工作流
- [ ] 文档编写

**周五**
- [ ] 项目演示
- [ ] 代码审查
- [ ] 后续改进建议

## 评估标准

### 技术深度
- [ ] 理解高级算法原理
- [ ] 实现正确的优化
- [ ] 达到性能目标

### 代码质量
- [ ] 模块化设计
- [ ] 错误处理
- [ ] 文档完整

### 创新性
- [ ] 独特的优化技巧
- [ ] 新的应用场景
- [ ] 扩展功能

## 总结

完成所有阶段后，您将：
1. **深入理解** FFT 算法的理论和实现
2. **熟练掌握** KISS FFT 库的使用和定制
3. **具备能力** 开发高效的信号处理应用
4. **拥有经验** 优化和扩展算法实现

## 进一步学习

- 研究其他 FFT 算法（如 Winograd FFT）
- 探索 GPU 加速（CUDA、OpenCL）
- 学习其他 DSP 算法（小波变换、滤波器设计）
- 参与开源 DSP 项目

---

恭喜完成整个学习计划！您现在是 FFT 专家了！🎉