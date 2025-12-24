# KISS FFT 数据类型切换指南

本文档详细说明如何在 KISS FFT 中切换和使用不同的数据类型。

## 目录

1. [数据类型概览](#数据类型概览)
2. [浮点数 vs 定点数](#浮点数-vs-定点数)
3. [编译时类型选择](#编译时类型选择)
4. [运行时类型切换](#运行时类型切换)
5. [性能对比](#性能对比)
6. [精度分析](#精度分析)
7. [最佳实践](#最佳实践)

---

## 数据类型概览

KISS FFT 支持以下数据类型：

| 类型 | 定义 | 范围 | 精度 | 适用场景 |
|------|------|------|------|----------|
| `float` | 单精度浮点 | ±3.4E38 | ~7 位小数 | 通用应用，实时系统 |
| `double` | 双精度浮点 | ±1.8E308 | ~15 位小数 | 高精度要求 |
| `int16_t` | 16位定点 | -32768~32767 | 整数 | 嵌入式，低功耗 |
| `int32_t` | 32位定点 | ±2.1E9 | 整数 | 高精度整数 |

---

## 浮点数 vs 定点数

### 浮点数 FFT

**优点：**
- 动态范围大
- 精度高
- 编程简单
- 现代CPU有硬件加速

**缺点：**
- 功耗较高（在某些平台）
- 不是所有嵌入式系统都支持
- 非确定性（某些实时系统）

**使用场景：**
- 桌面应用
- 服务器应用
- 高精度科学计算
- 音频处理

### 定点数 FFT

**优点：**
- 确定性执行时间
- 功耗低
- 适合无FPU的嵌入式
- 便于硬件实现

**缺点：**
- 动态范围受限
- 需要手动处理溢出
- 缩放操作复杂
- 精度较低

**使用场景：**
- 嵌入式系统
- DSP芯片
- 低功耗设备
- 实时控制系统

---

## 编译时类型选择

### 方法 1：使用编译宏（推荐）

#### 浮点数配置

```bash
# 单精度浮点（默认）
gcc -c kiss_fft.c -o kiss_fft_float.o

# 双精度浮点
gcc -DKISSFFT_DATATYPE=float -c kiss_fft.c -o kiss_fft_float.o
```

或者修改头文件：

```c
// kiss_fft.h 中添加
#ifndef KISSFFT_DATATYPE
#define KISSFFT_DATATYPE float
#endif

typedef KISSFFT_DATATYPE kiss_fft_scalar;
```

#### 定点数配置

```bash
# 16位定点数
gcc -DFIXED_POINT=16 -c kiss_fft.c -o kiss_fft_fixed16.o

# 32位定点数
gcc -DFIXED_POINT=32 -c kiss_fft.c -o kiss_fft_fixed32.o
```

### 方法 2：创建类型特定的配置文件

#### float_fft_config.h

```c
#ifndef FLOAT_FFT_CONFIG_H
#define FLOAT_FFT_CONFIG_H

#define KISSFFT_USE_FLOAT
typedef float kiss_fft_scalar;

// 浮点数不需要除法修正
#define C_FIXDIV(c, div)

#endif
```

#### fixed16_fft_config.h

```c
#ifndef FIXED16_FFT_CONFIG_H
#define FIXED16_FFT_CONFIG_H

#define FIXED_POINT 16
typedef int16_t kiss_fft_scalar;

// 定点数需要移位除法
#define C_FIXDIV(c, div) \
    do { \
        (c).r = (kiss_fft_scalar)(((int32_t)(c).r) >> (div)); \
        (c).i = (kiss_fft_scalar)(((int32_t)(c).i) >> (div)); \
    } while (0)

#endif
```

### 方法 3：创建多个库版本

```bash
#!/bin/bash
# build_all_variants.sh

echo "Building KISS FFT variants..."

# Float version
echo "Building float version..."
gcc -O2 -c kiss_fft.c -o kiss_fft_float.o
gcc -O2 -c kiss_fftr.c -o kiss_fftr_float.o
ar rcs libkissfft_float.a kiss_fft_float.o kiss_fftr_float.o

# Double version
echo "Building double version..."
gcc -DKISSFFT_DATATYPE=double -O2 -c kiss_fft.c -o kiss_fft_double.o
gcc -DKISSFFT_DATATYPE=double -O2 -c kiss_fftr.c -o kiss_fftr_double.o
ar rcs libkissfft_double.a kiss_fft_double.o kiss_fftr_double.o

# Fixed16 version
echo "Building fixed16 version..."
gcc -DFIXED_POINT=16 -O2 -c kiss_fft.c -o kiss_fft_fixed16.o
gcc -DFIXED_POINT=16 -O2 -c kiss_fftr.c -o kiss_fftr_fixed16.o
ar rcs libkissfft_fixed16.a kiss_fft_fixed16.o kiss_fftr_fixed16.o

# Fixed32 version
echo "Building fixed32 version..."
gcc -DFIXED_POINT=32 -O2 -c kiss_fft.c -o kiss_fft_fixed32.o
gcc -DFIXED_POINT=32 -O2 -c kiss_fftr.c -o kiss_fftr_fixed32.o
ar rcs libkissfft_fixed32.a kiss_fft_fixed32.o kiss_fftr_fixed32.o

echo "All variants built successfully!"
ls -lh libkissfft_*.a
```

---

## 运行时类型切换

### 方法 1：使用函数指针和多个库

```c
// fft_wrapper.h
typedef enum {
    FFT_TYPE_FLOAT,
    FFT_TYPE_DOUBLE,
    FFT_TYPE_FIXED16,
    FFT_TYPE_FIXED32
} FFTDataType;

typedef struct {
    FFTDataType type;
    void *cfg;          // 不透明的配置指针
    int nfft;

    // 函数指针
    void (*fft)(void *cfg, const void *fin, void *fout);
    void (*fft_free)(void *cfg);
} FFTContext;

// 创建指定类型的 FFT
FFTContext* fft_create(int nfft, int inverse, FFTDataType type);

// 执行 FFT
void fft_execute(FFTContext *ctx, const void *input, void *output);

// 销毁 FFT
void fft_destroy(FFTContext *ctx);
```

```c
// fft_wrapper.c
#include "fft_wrapper.h"
#include "kiss_fft.h"
#include <stdlib.h>

// Float 版本包装
typedef struct {
    kiss_fft_scalar r;
    kiss_fft_scalar i;
} cpx_float;

typedef struct {
    kiss_fft_cfg cfg;
    int nfft;
} float_fft_state;

void float_fft_wrapper(void *vctx, const void *vin, void *vout) {
    float_fft_state *ctx = (float_fft_state *)vctx;
    const cpx_float *in = (const cpx_float *)vin;
    cpx_float *out = (cpx_float *)vout;

    kiss_fft(ctx->cfg, (const kiss_fft_cpx *)in, (kiss_fft_cpx *)out);
}

FFTContext* fft_create(int nfft, int inverse, FFTDataType type) {
    FFTContext *ctx = malloc(sizeof(FFTContext));
    ctx->type = type;
    ctx->nfft = nfft;

    switch(type) {
        case FFT_TYPE_FLOAT: {
            float_fft_state *state = malloc(sizeof(float_fft_state));
            state->cfg = kiss_fft_alloc(nfft, inverse, NULL, NULL);
            state->nfft = nfft;
            ctx->cfg = state;
            ctx->fft = float_fft_wrapper;
            ctx->fft_free = (void (*)(void *))kiss_fft_free;
            break;
        }
        // 其他类型的实现...
        default:
            free(ctx);
            return NULL;
    }

    return ctx;
}

void fft_execute(FFTContext *ctx, const void *input, void *output) {
    ctx->fft(ctx->cfg, input, output);
}

void fft_destroy(FFTContext *ctx) {
    if (ctx) {
        ctx->fft_free(ctx->cfg);
        free(ctx);
    }
}
```

使用示例：

```c
#include "fft_wrapper.h"

int main() {
    // 选择数据类型
    FFTDataType type = FFT_TYPE_FLOAT;

    // 创建 FFT
    FFTContext *fft = fft_create(1024, 0, type);

    // 准备数据
    cpx_float input[1024];
    cpx_float output[1024];

    // 执行 FFT
    fft_execute(fft, input, output);

    // 清理
    fft_destroy(fft);

    return 0;
}
```

### 方法 2：使用联合体和类型标签

```c
// variant_fft.h
typedef union {
    float f;
    double d;
    int16_t i16;
    int32_t i32;
} fft_scalar;

typedef struct {
    fft_scalar r;
    fft_scalar i;
} fft_cpx;

typedef enum {
    SCALAR_FLOAT,
    SCALAR_DOUBLE,
    SCALAR_INT16,
    SCALAR_INT32
} scalar_type;

typedef struct {
    scalar_type type;
    int nfft;
    void *internal_state;
    void (*convert_to_internal)(const fft_cpx *src, void *dst, int n);
    void (*convert_from_internal)(const void *src, fft_cpx *dst, int n);
    void (*fft_func)(void *cfg, const void *fin, void *fout);
} variant_fft_cfg;
```

---

## 性能对比

### 基准测试代码

```c
// benchmark_types.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "kiss_fft.h"

#define N 1024
#define ITERATIONS 10000

// 定时宏
#define BENCHMARK_START() clock_t start = clock()
#define BENCHMARK_END(label) \
    do { \
        clock_t end = clock(); \
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC; \
        printf("%s: %.6f seconds (%.3f µs per FFT)\n", \
               label, elapsed, elapsed * 1000000 / ITERATIONS); \
    } while(0)

// Float 版本
void benchmark_float() {
    kiss_fft_cfg cfg = kiss_fft_alloc(N, 0, NULL, NULL);
    kiss_fft_cpx *input = malloc(sizeof(kiss_fft_cpx) * N);
    kiss_fft_cpx *output = malloc(sizeof(kiss_fft_cpx) * N);

    // 初始化
    for (int i = 0; i < N; i++) {
        input[i].r = (float)rand() / RAND_MAX;
        input[i].i = (float)rand() / RAND_MAX;
    }

    BENCHMARK_START();
    for (int i = 0; i < ITERATIONS; i++) {
        kiss_fft(cfg, input, output);
    }
    BENCHMARK_END("Float FFT");

    free(input);
    free(output);
    kiss_fft_free(cfg);
}

// 定点数版本（需要重新编译）
#ifdef FIXED_POINT
void benchmark_fixed16() {
    kiss_fft_cfg cfg = kiss_fft_alloc(N, 0, NULL, NULL);
    kiss_fft_cpx *input = malloc(sizeof(kiss_fft_cpx) * N);
    kiss_fft_cpx *output = malloc(sizeof(kiss_fft_cpx) * N);

    // 初始化
    for (int i = 0; i < N; i++) {
        input[i].r = rand() & 0xFFFF;
        input[i].i = rand() & 0xFFFF;
    }

    BENCHMARK_START();
    for (int i = 0; i < ITERATIONS; i++) {
        kiss_fft(cfg, input, output);
    }
    BENCHMARK_END("Fixed16 FFT");

    free(input);
    free(output);
    kiss_fft_free(cfg);
}
#endif

int main() {
    printf("KISS FFT Performance Benchmark\n");
    printf("================================\n");
    printf("FFT Size: %d\n", N);
    printf("Iterations: %d\n\n", ITERATIONS);

    benchmark_float();

#ifdef FIXED_POINT
    benchmark_fixed16();
#endif

    return 0;
}
```

### 典型性能对比

| 平台 | Float | Fixed16 | Fixed32 | 说明 |
|------|-------|---------|---------|------|
| x86_64 (有FPU) | 100% | 150% | 200% | 浮点最快 |
| ARM Cortex-M4 | 100% | 80% | 120% | 定点更快（有FPU） |
| ARM Cortex-M0 | N/A | 100% | 150% | 无FPU，定点唯一选择 |

---

## 精度分析

### 精度测试代码

```c
// precision_test.c
#include <stdio.h>
#include <math.h>
#include "kiss_fft.h"

#define PI 3.14159265358979323846
#define N 64

// 计算信噪比
double calculate_snr(const kiss_fft_cpx *signal, const kiss_fft_cpx *noise, int n) {
    double signal_power = 0;
    double noise_power = 0;

    for (int i = 0; i < n; i++) {
        double s_r = signal[i].r;
        double s_i = signal[i].i;
        double n_r = noise[i].r;
        double n_i = noise[i].i;

        signal_power += s_r * s_r + s_i * s_i;
        noise_power += n_r * n_r + n_i * n_i;
    }

    return 10 * log10(signal_power / noise_power);
}

// 测试 FFT 精度
void test_fft_precision() {
    kiss_fft_cfg cfg = kiss_fft_alloc(N, 0, NULL, NULL);
    kiss_fft_cpx *input = malloc(sizeof(kiss_fft_cpx) * N);
    kiss_fft_cpx *output = malloc(sizeof(kiss_fft_cpx) * N);
    kiss_fft_cpx *reconstructed = malloc(sizeof(kiss_fft_cpx) * N);
    kiss_fft_cfg icfg = kiss_fft_alloc(N, 1, NULL, NULL);

    // 创建测试信号：已知频率的正弦波
    for (int i = 0; i < N; i++) {
        double t = (double)i / N;
        input[i].r = cos(2 * PI * 5 * t);  // 5个周期
        input[i].i = sin(2 * PI * 5 * t);
    }

    // 正向 FFT
    kiss_fft(cfg, input, output);

    // 逆向 FFT
    kiss_fft(icfg, output, reconstructed);

    // 归一化并计算误差
    for (int i = 0; i < N; i++) {
        reconstructed[i].r /= N;
        reconstructed[i].i /= N;
    }

    // 计算重建信号与原信号的差异
    double max_error = 0;
    double mse = 0;
    for (int i = 0; i < N; i++) {
        double error_r = input[i].r - reconstructed[i].r;
        double error_i = input[i].i - reconstructed[i].i;
        double error = sqrt(error_r * error_r + error_i * error_i);

        if (error > max_error) max_error = error;
        mse += error * error;
    }
    mse /= N;

    printf("Precision Test Results:\n");
    printf("  Maximum error: %.10f\n", max_error);
    printf("  MSE: %.10f\n", mse);
    printf("  SNR: %.2f dB\n", 10 * log10(1.0 / mse));

    free(input);
    free(output);
    free(reconstructed);
    kiss_fft_free(cfg);
    kiss_fft_free(icfg);
}

int main() {
    test_fft_precision();
    return 0;
}
```

### 预期精度结果

| 类型 | SNR (dB) | 最大误差 | 说明 |
|------|----------|----------|------|
| float | ~120 dB | ~10^-6 | 单精度精度 |
| double | ~250 dB | ~10^-14 | 双精度精度 |
| int16_t | ~60 dB | ~10^-3 | 定点精度 |
| int32_t | ~120 dB | ~10^-6 | 高精度定点 |

---

## 最佳实践

### 1. 选择合适的类型

```c
// 决策树
if (需要高精度) {
    使用 double;
} else if (有FPU支持) {
    使用 float;  // 性能和精度的最佳平衡
} else if (数值范围小) {
    使用 int16_t;  // 省内存
} else {
    使用 int32_t;  // 大范围的整数运算
}
```

### 2. 输入数据缩放

```c
// 对于定点数，确保输入在合理范围
void scale_input_for_fixed_point(const float *input, int16_t *output, int n) {
    for (int i = 0; i < n; i++) {
        // 找到最大值
        float max_val = fabsf(input[i]);
        if (max_val > 1.0f) max_val = 1.0f;

        // 缩放到定点数范围
        output[i] = (int16_t)(input[i] * 32767.0f / max_val);
    }
}
```

### 3. 防止溢出

```c
// 定点数 FFT 的溢出检测
void check_overflow(const kiss_fft_cpx *data, int n) {
#ifdef FIXED_POINT
    for (int i = 0; i < n; i++) {
        if (data[i].r == 32767 || data[i].r == -32768 ||
            data[i].i == 32767 || data[i].i == -32768) {
            printf("Warning: Possible overflow at index %d\n", i);
        }
    }
#endif
}
```

### 4. 类型转换包装器

```c
// 通用的类型转换函数
typedef enum {
    CONVERT_FLOAT_TO_FIXED16,
    CONVERT_FIXED16_TO_FLOAT,
    // 更多转换...
} ConversionType;

void convert_samples(void *dst, const void *src, int n, ConversionType type) {
    switch(type) {
        case CONVERT_FLOAT_TO_FIXED16: {
            const float *s = (const float *)src;
            int16_t *d = (int16_t *)dst;
            for (int i = 0; i < n; i++) {
                float val = fmaxf(-1.0f, fminf(1.0f, s[i]));
                d[i] = (int16_t)(val * 32767.0f);
            }
            break;
        }
        case CONVERT_FIXED16_TO_FLOAT: {
            const int16_t *s = (const int16_t *)src;
            float *d = (float *)dst;
            for (int i = 0; i < n; i++) {
                d[i] = (float)s[i] / 32767.0f;
            }
            break;
        }
    }
}
```

---

## 实践练习

### 练习 1：编译多个版本

```bash
# 编译所有变体
./build_all_variants.sh

# 比较库文件大小
ls -lh libkissfft_*.a
```

### 练习 2：性能对比

```bash
# 编译并运行性能测试
gcc -O2 -o benchmark_float benchmark_types.c kiss_fft.c -lm
gcc -DFIXED_POINT=16 -O2 -o benchmark_fixed16 benchmark_types.c kiss_fft.c -lm

# 运行测试
./benchmark_float
./benchmark_fixed16
```

### 练习 3：精度测试

```bash
# 编译精度测试
gcc -o precision precision_test.c kiss_fft.c -lm

# 运行测试
./precision
```

---

## 总结

- KISS FFT 支持浮点数和定点数
- 编译时使用宏选择类型
- 定点数适合嵌入式系统
- 浮点数提供更高精度
- 可以创建运行时类型切换的包装器

**下一步：** 完成本阶段学习后，进入[阶段 3：实践应用](../phase3-practical/)。

---

**参考资源：**
- [项目 README](../../README.md)
- [定点数算法](https://en.wikipedia.org/wiki/Fixed-point_arithmetic)
- [IEEE 754 浮点数标准](https://en.wikipedia.org/wiki/IEEE_754)
